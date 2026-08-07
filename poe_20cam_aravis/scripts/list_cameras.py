#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/list_cameras.py
=======================
Print every camera Aravis can see, with the identity fields the app uses.

    python3 scripts/list_cameras.py
    ARV_FAKE_CAMERA_ENABLED=1 python3 scripts/list_cameras.py

The "log key" column is the filename the coverage CSV will use, so this is
also the quickest way to map a physical camera to its log file.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from poe_multi_aravis.app import _bootstrap_aravis   # noqa: E402

_bootstrap_aravis()

from poe_multi_aravis.aravis_backend.discovery import CameraDiscovery  # noqa: E402


def main() -> int:
    cameras = CameraDiscovery().refresh()
    if not cameras:
        print("No cameras found.")
        print("Hints: check the PoE link, set MTU 9000 and a link-local or "
              "static IP on the NIC, and confirm no other process holds the "
              "camera open.")
        return 1

    header = f"{'#':>3}  {'device id':<34} {'vendor':<14} {'model':<20} " \
             f"{'serial':<16} {'address':<16} log key"
    print(header)
    print("─" * len(header))
    for i, c in enumerate(cameras):
        print(f"{i:>3}  {c.device_id:<34} {(c.vendor or '-'):<14} "
              f"{(c.model or '-'):<20} {(c.serial_number or '-'):<16} "
              f"{(c.address or '-'):<16} {c.stable_key()}")
    print(f"\n{len(cameras)} camera(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
