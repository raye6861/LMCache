# SPDX-License-Identifier: Apache-2.0
"""GCU ops backend: forward the ``DeviceOps`` surface to ``triton_kernel_gcu``.

:class:`GcuDeviceOps` resolves the ``torch.ops.TRITON_GCU.lmcache.v0_5_5_*``
operators registered by the ``triton_kernel_gcu`` plugin and re-wraps them to
match the unified :class:`~lmcache.v1.platform.base.device_ops.DeviceOps`
surface. The signature translations performed here are:

1. ``paged_memory_device`` (a ``torch.device``) is reduced to the integer
   ``paged_device_index`` the Triton ops expect.
2. ``multi_layer_block_kv_transfer`` receives a ``PageBufferShapeDesc`` and
   expands it into the scalar ``kv_size/nl/nb/bs/nh/hs/element_size/
   block_stride_elems`` plus a ``dtype_code`` the Triton op expects.
3. ``multi_layer_kv_transfer`` forwards ``block_stride_elems`` (a 0.5.5
   addition accepted by the v0_5_5 kernel).

If ``triton_kernel_gcu`` is missing, a warning is logged and the instance stays
on the torch baseline (soft-fail, same as CUDA/XPU).
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, ClassVar
import json
import math

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.platform.base.device_ops import DeviceOps

logger = init_logger(__name__)


def _device_index(device: torch.device | str | None) -> int:
    """Return the integer device ordinal for *device*."""
    if device is None:
        return 0
    if not isinstance(device, torch.device):
        device = torch.device(device)
    return int(device.index or 0)


# ── 64-bit dtype handles that survive an "int64 → int32" remap ────────────
#
# Future vLLM-GCU may rebind ``torch.int64`` to the int32 dtype (indices and
# counts fit in 32 bits; pointer semantics move to uint64). In that setup
# ``torch.long`` still names the real 8-byte signed dtype, i.e.
# ``torch.int64 == torch.long`` becomes False. Resolve the handles that still
# name real 8-byte dtypes *by element size* here, so every dtype test,
# pointer-table read-back and pointer-table build below behaves identically
# with or without the remap -- no lmcache change is needed for either regime.


def _real_dtype(*handles: torch.dtype | None) -> torch.dtype | None:
    """Return the first handle that still names an 8-byte dtype."""
    for h in handles:
        if isinstance(h, torch.dtype) and torch.empty((), dtype=h).element_size() == 8:
            return h
    return None


_INT64 = _real_dtype(torch.int64, torch.long)  # real signed 64-bit
_UINT64 = _real_dtype(torch.uint64)            # real unsigned 64-bit

# Dtypes that may carry a pointer table. The 64-bit entries are spelled via
# the *real* handles: after the remap a real int64 table has
# ``dtype is torch.long`` while the remapped ``torch.int64`` is just int32.
_INT_DTYPES = tuple(
    dt for dt in (_INT64, torch.int32, _UINT64, torch.uint32) if dt is not None
)
_REAL64_DTYPES = tuple(dt for dt in (_INT64, _UINT64) if dt is not None)


def _pointer_table_values(values: torch.Tensor) -> list[int]:
    """Read an already-built pointer table back as a list of 64-bit values.

    Never read 64-bit tensors through ``.cpu()``/``.tolist()`` (GCU splits
    them into int32 halves) and never ``.to(torch.int64)`` (int64 may be
    remapped to a 4-byte dtype in future vLLM-GCU): the int32 view is the only
    width-independent observation, valid on CPU and on GCU, for both the
    int64 and uint64 spellings of a table.
    """
    t = values.detach()
    if t.dtype in _REAL64_DTYPES:
        # int32 readback is *signed*: a hi half with bit31 set (pointers
        # >= 2**63, e.g. a 0xFFFF.... mapping) comes back as a negative
        # Python int (-1 for 0xFFFFFFFF). Mask to the unsigned bit pattern
        # so the reassembled value matches data_ptr()'s unsigned convention;
        # without it, _pointer_table_values would return
        # 0xFFFFFFFF11223344 - 2**64 for such a pointer.
        lo_hi = [
            int(v) & 0xFFFFFFFF
            for v in t.reshape(-1).view(torch.int32).cpu().tolist()
        ]
        return [lo_hi[i] | (lo_hi[i + 1] << 32) for i in range(0, len(lo_hi), 2)]
    return [int(v) for v in t.cpu().reshape(-1).tolist()]


def _extract_pointer_values(
    values: torch.Tensor | list[int] | list[torch.Tensor],
) -> list[int]:
    """Normalize the upstream input shapes into a list of 64-bit values.

    The three shapes map 1:1 to how the call sites invoke the transfer ops:

    1. ``torch.Tensor`` with an integer dtype -- an already-built pointer
       table (e.g. the uint64 ``kv_cache_pointers`` the GCU connectors build):
       take its *contents*.
    2. ``torch.Tensor`` with a non-integer dtype -- a memory object (e.g. a
       fused cross-layer KV tensor): take its ``data_ptr()``. Converting its
       contents would feed KV *data* as a pointer table.
    3. ``list[torch.Tensor]`` / ``list[int]`` -- memory objects / raw host
       pointers: ``data_ptr()`` of each / the values themselves.
    """
    if isinstance(values, torch.Tensor):
        if values.dtype in _INT_DTYPES:
            return _pointer_table_values(values)
        return [int(values.data_ptr())]
    if values and isinstance(values[0], torch.Tensor):
        return [int(t.data_ptr()) for t in values]
    return [int(v) for v in values]


def _as_cpu_int64_tensor(
    values: torch.Tensor | list[int] | list[torch.Tensor],
) -> torch.Tensor:
    """Build a CPU pointer table (required by GCU KV kernels).

    GCU cannot load full 64-bit values from an on-device tensor (only the low
    32 bits survive), so pointer tables must stay on CPU; the Triton kernels
    split them into lo/hi int32 on the host.

    The table is stored as ``uint64`` (was ``int64``): pointer values are the
    one place a 64-bit value must survive byte-exact, uint64 is unaffected
    when vLLM-GCU rebinds ``torch.int64`` to a 4-byte dtype, it holds every
    address without two's-complement wrapping, and the shipped
    ``triton_kernel_gcu`` accepts CPU uint64 tables.
    """
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().to(torch.uint64).contiguous()
    if values and isinstance(values[0], torch.Tensor):
        return torch.tensor(
            [int(t.data_ptr()) for t in values], dtype=torch.uint64, device="cpu"
        )
    return torch.tensor(list(values), dtype=torch.uint64, device="cpu")


def _as_gcu_int64_tensor(
    values: torch.Tensor | list[int] | list[torch.Tensor],
    device: torch.device | str | int | None = None,
) -> torch.Tensor:
    """Build a GCU ``uint64`` pointer table with *true* 64-bit storage.

    Same input shapes as :func:`_as_cpu_int64_tensor`, but the result is a
    device tensor instead of a CPU one. Directly constructing an int64/uint64
    tensor on GCU is silently truncated to int32 storage (only the low 32
    bits survive), so the only construction path that yields a true 64-bit
    device table is to upload the values as a flat uint32 [lo, hi] table and
    ``view(torch.uint64)`` -- a pure metadata op that aliases real 8-byte
    pairs. Host-side reads of the result are still mangled (64-bit D2H is
    split into int32); read back via ``tab.view(torch.int32).cpu()``.
    """
    if device is None:
        dev = torch.device("gcu", 0)
    elif isinstance(device, int):
        dev = torch.device("gcu", device)
    else:
        dev = torch.device(device)
        if dev.type == "gcu" and dev.index is None:
            dev = torch.device("gcu", 0)

    if _UINT64 is None:
        raise RuntimeError(
            "No real 8-byte unsigned dtype is available; cannot build a "
            "true 64-bit GCU pointer table."
        )

    vals = _extract_pointer_values(values)

    flat: list[int] = []
    for v in vals:
        # Little-endian [lo, hi] pair per 64-bit value: lo sits at the lower
        # address so that a native (little-endian) u64 load of the pair
        # reassembles ``v`` exactly.
        flat.append(v & 0xFFFFFFFF)
        flat.append((v >> 32) & 0xFFFFFFFF)
    tab_u32 = torch.tensor(flat, dtype=torch.uint32, device=dev)
    return tab_u32.view(_UINT64)


def _normalize_slot_mapping(
    slot_mapping: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """GCU kernels expect ``int32`` slot indices on the KV tensor device."""
    if slot_mapping.dtype != torch.int32 or slot_mapping.device != device:
        return slot_mapping.to(
            device=device, dtype=torch.int32, copy=False
        ).contiguous()
    if not slot_mapping.is_contiguous():
        return slot_mapping.contiguous()
    return slot_mapping


def _normalize_rope_positions(
    positions: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """GCU rope kernels expect contiguous ``int32`` positions on device."""
    if positions.dtype != torch.int32 or positions.device != device:
        return positions.to(device=device, dtype=torch.int32, copy=False).contiguous()
    if not positions.is_contiguous():
        return positions.contiguous()
    return positions


def _dtype_to_code(dtype: torch.dtype | None) -> int:
    """Encode a KV dtype into the integer code the v0_5_5 kernels accept.

    Must stay in sync with ``triton_kernel_gcu.kernels.lmcache.cpu_ref.
    kv_transfer_v0_5_5._dtype_from_code``: 0=bfloat16, 1=float16, 2=float32,
    3=uint8.
    """
    if dtype == torch.float16:
        return 1
    if dtype == torch.float32:
        return 2
    if dtype == torch.uint8:
        return 3
    return 0


def _tensor_from_gcu_ptr(
    ptr: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create a non-owning GCU tensor view over a raw device pointer.

    GCU has no ``_construct_storage_from_data_pointer`` equivalent, so we use
    the ``__cuda_array_interface__`` round-trip that torch_gcu itself uses:
    wrap the pointer as a flat ``uint8`` array and view it as the target dtype.
    The returned tensor aliases ``ptr``, so the caller must keep the
    underlying allocation alive.
    """

    class _GcuArrayWrapper:
        def __init__(self, ptr_int: int, n: int) -> None:
            self.__cuda_array_interface__ = {
                "data": (ptr_int, False),
                "shape": (n,),
                "typestr": "|u1",
                "version": 3,
            }

    numel = math.prod(shape)
    total_bytes = numel * dtype.itemsize
    flat = torch.as_tensor(_GcuArrayWrapper(ptr, total_bytes), device=device)
    return flat.view(dtype).reshape(*shape)


class GcuDeviceOps(DeviceOps):
    """GCU ops backend assembling its surface from ``triton_kernel_gcu``."""

    device_type: ClassVar[str] = "gcu"

    def __init__(self) -> None:
        super().__init__()
        self._ns: Any | None = None

    def tensor_from_ptr(
        self,
        ptr: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Create a GCU tensor from a process-local device pointer."""
        return _tensor_from_gcu_ptr(ptr, shape, dtype, device)

    def ensure_native(self) -> None:
        """Load ``triton_kernel_gcu`` and cache its lmcache op namespace."""
        if self._native_bound:
            return
        self._native_bound = True  # set early to prevent repeated attempts
        try:
            # Third Party
            import triton_kernel_gcu  # noqa: F401
        except ImportError:
            logger.warning(
                "triton_kernel_gcu plugin not found; GcuDeviceOps stays on "
                "the torch baseline for all ops."
            )
            return
        self._ns = torch.ops.TRITON_GCU.lmcache

    def _op(self, suffix: str) -> Any:
        """Resolve ``v0_5_5_<suffix>`` on the TRITON_GCU lmcache namespace."""
        if self._ns is None:
            raise RuntimeError(
                "triton_kernel_gcu plugin is not loaded; GCU ops are unavailable."
            )
        name = f"v0_5_5_{suffix}"
        try:
            return getattr(self._ns, name)
        except AttributeError as exc:
            raise RuntimeError(
                f"Missing torch.ops.TRITON_GCU.lmcache.{name}; "
                "ensure triton_kernel_gcu finished registration."
            ) from exc

    # ── Ops: KV transfer ─────────────────────────────────────────────

    def multi_layer_kv_transfer(
        self,
        key_value: torch.Tensor,
        key_value_ptrs: torch.Tensor | list[torch.Tensor],
        slot_mapping: torch.Tensor,
        paged_memory_device: torch.device | str,
        page_buffer_size: int,
        direction: Any,
        engine_kv_format: Any,
        block_size: int = 0,
        head_size: int = 0,
        skip_prefix_n_tokens: int = 0,
        block_stride_elems: int = 0,
    ) -> None:
        ptrs = _as_cpu_int64_tensor(key_value_ptrs)
        slots = _normalize_slot_mapping(slot_mapping, key_value.device)
        self._op("multi_layer_kv_transfer")(
            key_value,
            ptrs,
            slots,
            _device_index(paged_memory_device),
            page_buffer_size,
            int(direction),
            int(engine_kv_format),
            block_size,
            head_size,
            skip_prefix_n_tokens,
            block_stride_elems,
        )

    def single_layer_kv_transfer(
        self,
        lmc_key_value_cache: torch.Tensor,
        vllm_key_value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        direction: Any,
        engine_kv_format: Any,
        token_major: bool = False,
    ) -> None:
        slots = _normalize_slot_mapping(slot_mapping, lmc_key_value_cache.device)
        self._op("single_layer_kv_transfer")(
            lmc_key_value_cache,
            vllm_key_value_cache,
            slots,
            int(direction),
            int(engine_kv_format),
            token_major,
        )

    def multi_layer_block_kv_transfer(
        self,
        paged_buffer_ptrs_tensor: torch.Tensor | list,
        # Declared ``list[int]`` (not a tensor list) so the MP gather/scatter
        # detection routes through address conversion: the GCU Triton kernel
        # reads each object's *host* address as a raw pointer, which requires
        # pinned (registered) host memory that the address path guarantees.
        lmcache_objects_ptrs: list[int],
        block_ids: torch.Tensor | list[int],
        device: torch.device | str,
        direction: Any,
        shape_desc: Any,
        lmcache_chunk_size: int,
        engine_kv_format: Any,
        skip_prefix_n_blocks: int,
    ) -> None:
        dev = device if isinstance(device, torch.device) else torch.device(device)
        if isinstance(paged_buffer_ptrs_tensor, torch.Tensor):
            paged_ptrs = paged_buffer_ptrs_tensor
        else:
            paged_ptrs = torch.tensor(
                list(paged_buffer_ptrs_tensor), dtype=torch.uint64, device=dev
            )
        object_ptrs = torch.tensor(lmcache_objects_ptrs, dtype=torch.uint64, device=dev)
        if isinstance(block_ids, torch.Tensor):
            blocks = block_ids.to(device=dev, copy=False)
        else:
            blocks = torch.tensor(block_ids, device=dev)
        dtype_code = _dtype_to_code(getattr(shape_desc, "dtype", None))
        # Pin the calling thread's current device for the launch. torch.gcu
        # tracks the current device per thread, and the Triton GCU backend
        # launches on the thread-local current device (not on the tensors'
        # device). A thread whose current device is still 0 otherwise fails
        # TOPS launch validation when the KV tensors live on another GCU
        # (e.g. TP1's gcu:1), so route the launch through the real device.
        with torch_dev.device(dev):
            self._op("multi_layer_block_kv_transfer")(
                paged_ptrs,
                object_ptrs,
                blocks,
                _device_index(device),
                int(direction),
                shape_desc.kv_size,
                shape_desc.nl,
                shape_desc.nb,
                shape_desc.bs,
                shape_desc.nh,
                shape_desc.hs,
                shape_desc.element_size,
                shape_desc.block_stride_elems,
                lmcache_chunk_size,
                int(engine_kv_format),
                skip_prefix_n_blocks,
                dtype_code,
            )

    def load_and_reshape_flash(
        self,
        key_value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer_idx: int,
    ) -> None:
        slots = _normalize_slot_mapping(slot_mapping, key_value.device)
        self._op("load_and_reshape_flash")(
            key_value, key_cache, value_cache, slots, layer_idx
        )

    def reshape_and_cache_back_flash(
        self,
        key_value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer_idx: int,
    ) -> None:
        slots = _normalize_slot_mapping(slot_mapping, key_value.device)
        self._op("reshape_and_cache_back_flash")(
            key_value, key_cache, value_cache, slots, layer_idx
        )

    # ── Ops: codec ───────────────────────────────────────────────────

    def calculate_cdf(self, input_tensor: torch.Tensor, num_bins: int) -> torch.Tensor:
        return self._op("calculate_cdf")(input_tensor, int(num_bins))

    def encode_fast_new(
        self,
        cdf: torch.Tensor,
        input_sym: torch.Tensor,
        output_buffer: torch.Tensor,
        output_lengths: torch.Tensor,
    ) -> None:
        self._op("encode_fast_new")(cdf, input_sym, output_buffer, output_lengths)

    def decode_fast_new(
        self,
        cdf: torch.Tensor,
        bytestreams: torch.Tensor,
        lengths: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        self._op("decode_fast_new")(cdf, bytestreams, lengths, output)

    def decode_fast_prefsum(
        self,
        cdf: torch.Tensor,
        bytestreams: torch.Tensor,
        lengths_prefsum: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        self._op("decode_fast_prefsum")(cdf, bytestreams, lengths_prefsum, output)

    # ── Ops: rotary ──────────────────────────────────────────────────

    def rotary_embedding_k_fused(
        self,
        old_positions: torch.Tensor,
        new_positions: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
    ) -> None:
        device = key.device
        old_pos = _normalize_rope_positions(old_positions, device)
        new_pos = _normalize_rope_positions(new_positions, device)
        self._op("rotary_embedding_k_fused")(
            old_pos, new_pos, key, int(head_size), cos_sin_cache, is_neox
        )

    def rotary_embedding_k_fused_strided(
        self,
        old_positions: torch.Tensor,
        new_positions: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        head_stride: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
    ) -> None:
        device = key.device
        old_pos = _normalize_rope_positions(old_positions, device)
        new_pos = _normalize_rope_positions(new_positions, device)
        self._op("rotary_embedding_k_fused_strided")(
            old_pos,
            new_pos,
            key,
            int(head_size),
            int(head_stride),
            cos_sin_cache,
            is_neox,
        )

    # ── Ops: async memcpy ────────────────────────────────────────────

    def lmcache_memcpy_async(
        self,
        dest: Any,
        src: Any,
        nbytes: int,
        direction: Any,
        host_buffer_offset: int,
        host_buffer_alignments: int,
    ) -> None:
        if isinstance(dest, torch.Tensor) or isinstance(src, torch.Tensor):
            super().lmcache_memcpy_async(
                dest, src, nbytes, direction, host_buffer_offset, host_buffer_alignments
            )
            return
        self._op("lmcache_memcpy_async")(
            int(dest),
            int(src),
            int(nbytes),
            int(direction),
            int(host_buffer_offset),
            int(host_buffer_alignments),
        )

    # ── Ops: batched memcpy ──────────────────────────────────────────

    def batched_memcpy(
        self,
        src_ptrs: Any,
        dst_ptrs: Any,
        sizes: Any,
    ) -> None:
        self._op("batched_memcpy")(src_ptrs, dst_ptrs, sizes)

    # ── Ops: event / completion recording ────────────────────────────

    def record_completion_on_stream(
        self,
        stream_ptr: int,
        kind: str,
        payload: bytes,
    ) -> None:
        # The GCU host op round-trips the payload through a JSON string, so
        # the msgpack bytes are latin1-decoded (lossless) into a ``str``.
        self._op("record_completion_on_stream")(
            int(stream_ptr), kind, payload.decode("latin1")
        )

    def drain_recorded_completions(self) -> list[tuple[str, bytes]]:
        raw = self._op("drain_recorded_completions")()
        if not raw:
            return []
        return [(kind, payload.encode("latin1")) for kind, payload in json.loads(raw)]

    def record_event_on_stream(
        self,
        stream_ptr: int,
        event_type_name: str,
        session_id: str,
        str_metadata: dict[str, str],
        int_metadata: dict[str, int],
    ) -> None:
        self._op("record_event_on_stream")(
            int(stream_ptr), event_type_name, session_id, str_metadata, int_metadata
        )

    def drain_recorded_events(
        self,
    ) -> list[tuple[str, str, float, dict[str, str], dict[str, int]]]:
        raw = self._op("drain_recorded_events")()
        if not raw:
            return []
        return [tuple(item) for item in json.loads(raw)]

    # ── Ops: host alloc / free ───────────────────────────────────────

    def alloc_pinned_ptr(self, size: int, device_id: int = 0) -> int:
        return int(self._op("alloc_pinned_ptr")(size, device_id))

    def free_pinned_ptr(self, ptr: int) -> None:
        self._op("free_pinned_ptr")(ptr)

    def alloc_pinned_numa_ptr(self, size: int, numa_id: int = 0) -> int:
        return int(self._op("alloc_pinned_numa_ptr")(size, numa_id))

    def free_pinned_numa_ptr(self, ptr: int, size: int | None = None) -> None:
        self._op("free_pinned_numa_ptr")(ptr, size or 0)

    def alloc_hugepage_pinned_ptr(self, size: int, device_id: int = 0) -> int:
        return int(self._op("alloc_hugepage_pinned_ptr")(size, device_id))

    def free_hugepage_pinned_ptr(self, ptr: int, size: int = 0) -> None:
        self._op("free_hugepage_pinned_ptr")(ptr, size)

    def alloc_hugepage_pinned_numa_ptr(self, size: int, numa_id: int = 0) -> int:
        return int(self._op("alloc_hugepage_pinned_numa_ptr")(size, numa_id))

    def free_hugepage_pinned_numa_ptr(self, ptr: int, size: int = 0) -> None:
        self._op("free_hugepage_pinned_numa_ptr")(ptr, size)

    def alloc_numa_ptr(self, size: int, numa_id: int = 0) -> int:
        return int(self._op("alloc_numa_ptr")(size, numa_id))

    def free_numa_ptr(self, ptr: int, size: int | None = None) -> None:
        self._op("free_numa_ptr")(ptr, size or 0)

    def alloc_shm_pinned_ptr(self, size: int, shm_name: str = "") -> int:
        return int(self._op("alloc_shm_pinned_ptr")(size, shm_name))

    def free_shm_pinned_ptr(self, ptr: int, size: int = 0, shm_name: str = "") -> None:
        self._op("free_shm_pinned_ptr")(ptr, size, shm_name)

    # ── Ops: misc ────────────────────────────────────────────────────

    def get_gpu_pci_bus_id(self, device_id: int = 0) -> str | None:
        bus = self._op("get_gpu_pci_bus_id")(int(device_id))
        if not bus:
            return None
        return str(bus)
