# SPDX-License-Identifier: Apache-2.0
"""GCU device-event IPC backend.

GCU's ``torch.gcu.Event`` exposes the CUDA-style interprocess API
(``Event(interprocess=True)``, ``ipc_handle()``, ``from_ipc_handle()``). Its C
extension kwlist is not reflected by :func:`inspect.signature`, so the generic
capability probe in :class:`DefaultEventIPCBackend` reports a false negative.
This backend overrides the capability check to probe by direct construction
while reusing the generic event operations.
"""

# Future
from __future__ import annotations

# Standard
import os

# Third Party
import torch

# First Party
from lmcache.v1.platform.base.event_ipc import DefaultEventIPCBackend


class GcuEventIPCBackend(DefaultEventIPCBackend):
    """Event IPC backend backed by GCU interprocess events."""

    def __init__(self) -> None:
        """Bind the backend to the ``torch.gcu`` event module."""
        super().__init__(event_module=torch.gcu, device_type="gcu")

    def check_event_support(self, device: object) -> None:
        """Validate GCU interprocess event support by direct probe.

        The generic probe relies on :func:`inspect.signature`, which cannot
        see the GCU C extension's ``interprocess`` kwarg even though the
        ``Event`` docstring documents it. Construct an interprocess event
        directly to fail closed on the real capability.

        Args:
            device: Target GCU device (used only for error reporting).

        Raises:
            RuntimeError: If ``torch.gcu.Event`` is missing, does not accept
                ``interprocess=True``, or lacks ``from_ipc_handle``.
        """
        event_cls = getattr(torch.gcu, "Event", None)
        if event_cls is None:
            raise RuntimeError(
                "Device backend 'gcu' does not support interprocess events: "
                "no Event class is available."
            )
        try:
            event_cls(interprocess=True)
        except TypeError as exc:
            raise RuntimeError(
                "Device backend 'gcu' does not support interprocess events: "
                "Event does not accept interprocess=True."
            ) from exc
        if not hasattr(event_cls, "from_ipc_handle"):
            raise RuntimeError(
                "Device backend 'gcu' does not support interprocess event "
                "handles: Event.from_ipc_handle is missing."
            )

    # TEMP DIAGNOSTIC: isolate whether a hang is caused by GCU's
    # cross-process event primitive (Event.from_ipc_handle / Event.wait).
    # With LMCACHE_GCU_SKIP_EVENT_IPC=1 the import returns None and the wait
    # becomes a no-op, so both store and retrieve proceed without the
    # cross-process event barrier. Revert by unsetting the env var.
    def import_event(self, handle: bytes, device: object) -> object:
        """Import a cross-process event, or return ``None`` when disabled.

        Args:
            handle: Serialized event handle from another process.
            device: Target GCU device.

        Returns:
            The imported event, or ``None`` when ``LMCACHE_GCU_SKIP_EVENT_IPC``
            is set to ``"1"``.
        """
        if os.environ.get("LMCACHE_GCU_SKIP_EVENT_IPC") == "1":
            return None
        return super().import_event(handle, device)

    def wait_event(self, event: object, stream: object) -> None:
        """Make ``stream`` wait for ``event``, or no-op when disabled.

        Args:
            event: Backend-native event to wait for.
            stream: Stream that should wait for the event.
        """
        if os.environ.get("LMCACHE_GCU_SKIP_EVENT_IPC") == "1":
            return
        super().wait_event(event, stream)
