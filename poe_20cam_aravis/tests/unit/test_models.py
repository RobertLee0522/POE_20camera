# -*- coding: utf-8 -*-
"""Domain model behaviour — mainly the camera identity rules."""

from poe_multi_aravis.domain.models import CameraDescriptor, StreamStatistics


def test_display_name_prefers_ip():
    d = CameraDescriptor(device_id="Hik-abc", vendor="Hikrobot",
                         model="MV-CA050", serial_number="SN123",
                         address="192.168.1.11")
    assert d.display_name() == "MV-CA050  [192.168.1.11]"


def test_display_name_falls_back_to_serial_then_model():
    d = CameraDescriptor(device_id="usb-1", vendor="V", model="M",
                         serial_number="SN9")
    assert d.display_name() == "M  S/N:SN9"
    assert CameraDescriptor(device_id="x", model="M").display_name() == "M"
    assert CameraDescriptor(device_id="raw-id").display_name() == "raw-id"


def test_stable_key_prefers_serial_and_is_filesystem_safe():
    d = CameraDescriptor(device_id="a/b", serial_number="SN 12:34",
                         address="192.168.1.11")
    assert d.stable_key() == "SN_12_34"


def test_stable_key_uses_address_then_device_id():
    assert CameraDescriptor(device_id="x", address="10.0.0.5").stable_key() == "10.0.0.5"
    assert CameraDescriptor(device_id="dev/1").stable_key() == "dev_1"


def test_stream_statistics_reset():
    s = StreamStatistics(frames_completed=10, acquisition_fps=12.5)
    s.reset()
    assert s.frames_completed == 0
    assert s.acquisition_fps == 0.0
