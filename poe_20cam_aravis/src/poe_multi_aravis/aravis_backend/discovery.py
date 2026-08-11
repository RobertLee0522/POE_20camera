# -*- coding: utf-8 -*-
"""
aravis_backend/discovery.py
===========================
Camera discovery using the Aravis 0.10 GObject Introspection API.

Discovery is deliberately kept separate from connection: enumerating
20 cameras must never open 20 device handles.  Only metadata already
collected by ``Aravis.update_device_list()`` is read here.
"""

from __future__ import annotations

import logging
import os
from typing import List

import gi
gi.require_version("Aravis", "0.10")
from gi.repository import Aravis

from ..domain.models import CameraDescriptor

log = logging.getLogger(__name__)

# Allow the Aravis fake camera in tests / demo mode
_FAKE_ENABLED = os.environ.get("ARV_FAKE_CAMERA_ENABLED", "0") == "1"


def _ensure_fake_interface() -> None:
    """Enable the Aravis fake-camera interface if requested."""
    if _FAKE_ENABLED:
        try:
            Aravis.enable_interface("Fake")
        except Exception:
            pass


class CameraDiscovery:
    """
    Discovers GenICam-compatible cameras via Aravis.

    ``refresh()`` never opens a camera connection, which matters a great
    deal with 20 PoE cameras on one switch: opening every device just to
    build a list would stall the UI and hold control channels open.
    """

    def refresh(self) -> List[CameraDescriptor]:
        """
        Update the Aravis device list and return all discovered cameras.

        Returns:
            List of CameraDescriptor, ordered as Aravis reports them.
            May be empty.
        """
        _ensure_fake_interface()
        try:
            Aravis.update_device_list()
        except Exception as exc:
            log.error("Aravis device enumeration failed: %s", exc)
            return []

        count = Aravis.get_n_devices()
        descriptors: List[CameraDescriptor] = []

        for i in range(count):
            try:
                device_id = Aravis.get_device_id(i)
            except Exception:
                continue

            vendor   = _safe_str(lambda: Aravis.get_device_vendor(i))
            model    = _safe_str(lambda: Aravis.get_device_model(i))
            serial   = _safe_str(lambda: Aravis.get_device_serial_nbr(i))
            protocol = _safe_str(lambda: Aravis.get_device_protocol(i))
            address  = _safe_str(lambda: Aravis.get_device_address(i))

            descriptors.append(CameraDescriptor(
                device_id=device_id,
                vendor=vendor,
                model=model,
                serial_number=serial,
                transport=protocol,
                address=address,
            ))
            log.debug("Discovered [%02d]: %s (vendor=%s model=%s sn=%s ip=%s)",
                      i, device_id, vendor, model, serial, address)

        log.info("Discovery complete: %d camera(s) found", len(descriptors))
        return descriptors


def _safe_str(fn) -> "str | None":
    try:
        v = fn()
        return str(v).strip() if v else None
    except Exception:
        return None
