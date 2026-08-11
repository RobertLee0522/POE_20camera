# -*- coding: utf-8 -*-
"""Camera state machine transitions."""

import pytest

from poe_multi_aravis.domain.camera_state import CameraState, CameraStateMachine


def test_happy_path_to_streaming():
    sm = CameraStateMachine()
    assert sm.state == CameraState.DISCONNECTED
    sm.transition(CameraState.CONNECTING)
    sm.transition(CameraState.CONNECTED)
    assert sm.can_start()
    sm.transition(CameraState.STARTING)
    sm.transition(CameraState.STREAMING)
    assert sm.is_streaming()
    assert sm.can_stop()


def test_invalid_transition_raises():
    sm = CameraStateMachine()
    with pytest.raises(ValueError):
        sm.transition(CameraState.STREAMING)


def test_force_error_and_recover():
    sm = CameraStateMachine()
    sm.transition(CameraState.CONNECTING)
    sm.force_error()
    assert sm.state == CameraState.ERROR
    assert sm.can_connect()
    sm.transition(CameraState.CONNECTING)     # ERROR → CONNECTING is allowed


def test_listeners_receive_transitions():
    seen = []
    sm = CameraStateMachine()
    sm.add_listener(lambda old, new: seen.append((old, new)))
    sm.transition(CameraState.CONNECTING)
    sm.transition(CameraState.CONNECTED)
    assert seen == [
        (CameraState.DISCONNECTED, CameraState.CONNECTING),
        (CameraState.CONNECTING, CameraState.CONNECTED),
    ]


def test_listener_exception_does_not_break_transition():
    sm = CameraStateMachine()

    def boom(_old, _new):
        raise RuntimeError("listener failed")

    sm.add_listener(boom)
    sm.transition(CameraState.CONNECTING)
    assert sm.state == CameraState.CONNECTING
