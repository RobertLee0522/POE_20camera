# -*- coding: utf-8 -*-
"""
domain/camera_state.py
======================
Camera state machine with explicit valid transitions.
The UI derives button availability from state rather than
manually enabling/disabling controls in callbacks.
"""

from __future__ import annotations

from enum import Enum, auto
from threading import Lock
from typing import Optional, Callable, Set


class CameraState(Enum):
    DISCONNECTED = auto()
    DISCOVERED   = auto()
    CONNECTING   = auto()
    CONNECTED    = auto()
    STARTING     = auto()
    STREAMING    = auto()
    STOPPING     = auto()
    ERROR        = auto()


# Valid transitions: {from_state: {to_states}}
_VALID_TRANSITIONS: dict[CameraState, Set[CameraState]] = {
    CameraState.DISCONNECTED: {CameraState.DISCOVERED, CameraState.CONNECTING},
    CameraState.DISCOVERED:   {CameraState.CONNECTING, CameraState.DISCONNECTED},
    CameraState.CONNECTING:   {CameraState.CONNECTED, CameraState.ERROR, CameraState.DISCONNECTED},
    CameraState.CONNECTED:    {CameraState.STARTING, CameraState.DISCONNECTED, CameraState.ERROR},
    CameraState.STARTING:     {CameraState.STREAMING, CameraState.ERROR, CameraState.CONNECTED},
    CameraState.STREAMING:    {CameraState.STOPPING, CameraState.ERROR},
    CameraState.STOPPING:     {CameraState.CONNECTED, CameraState.ERROR, CameraState.DISCONNECTED},
    CameraState.ERROR:        {CameraState.DISCONNECTED, CameraState.CONNECTING},
}


class CameraStateMachine:
    """
    Thread-safe state machine for the camera session lifecycle.
    Raises ValueError on invalid transitions.
    """

    def __init__(self) -> None:
        self._state = CameraState.DISCONNECTED
        self._lock = Lock()
        self._listeners: list[Callable[[CameraState, CameraState], None]] = []

    # ── public API ────────────────────────────────────────────

    @property
    def state(self) -> CameraState:
        with self._lock:
            return self._state

    def transition(self, new_state: CameraState) -> None:
        """Attempt a state transition. Raises ValueError if invalid."""
        with self._lock:
            old = self._state
            allowed = _VALID_TRANSITIONS.get(old, set())
            if new_state not in allowed:
                raise ValueError(
                    f"Invalid state transition: {old.name} → {new_state.name}"
                )
            self._state = new_state

        # Notify listeners outside the lock
        for cb in list(self._listeners):
            try:
                cb(old, new_state)
            except Exception:
                pass

    def force_error(self) -> None:
        """Force transition to ERROR from any active state."""
        with self._lock:
            self._state = CameraState.ERROR
        for cb in list(self._listeners):
            try:
                cb(CameraState.DISCONNECTED, CameraState.ERROR)
            except Exception:
                pass

    def reset(self) -> None:
        """Force transition to DISCONNECTED (used during cleanup)."""
        with self._lock:
            self._state = CameraState.DISCONNECTED

    def add_listener(self, cb: Callable[[CameraState, CameraState], None]) -> None:
        self._listeners.append(cb)

    def remove_listener(self, cb: Callable[[CameraState, CameraState], None]) -> None:
        try:
            self._listeners.remove(cb)
        except ValueError:
            pass

    # ── derived UI helpers ────────────────────────────────────

    def can_connect(self) -> bool:
        return self.state in (CameraState.DISCONNECTED, CameraState.DISCOVERED, CameraState.ERROR)

    def can_disconnect(self) -> bool:
        return self.state in (CameraState.CONNECTED,)

    def can_start(self) -> bool:
        return self.state == CameraState.CONNECTED

    def can_stop(self) -> bool:
        return self.state == CameraState.STREAMING

    def is_streaming(self) -> bool:
        return self.state == CameraState.STREAMING

    def is_connected_or_streaming(self) -> bool:
        return self.state in (CameraState.CONNECTED, CameraState.STARTING,
                               CameraState.STREAMING, CameraState.STOPPING)
