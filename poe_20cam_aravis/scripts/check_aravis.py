#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/check_aravis.py
=======================
Environment sanity check.  Run this first when something will not start:

    python3 scripts/check_aravis.py

It reports whether PyGObject, the Aravis typelib, PyQt5, OpenCV, NumPy,
psutil, matplotlib, PyYAML and (optionally) CUDA are importable, and
prints the Aravis version plus the number of visible cameras.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

OK, BAD, WARN = "  ✓", "  ✗", "  !"


def check_aravis() -> bool:
    try:
        from poe_multi_aravis.app import _bootstrap_aravis
        _bootstrap_aravis()
        import gi
        from gi.repository import Aravis
    except Exception as exc:
        print(f"{BAD} Aravis bindings: {exc}")
        print("      → install libaravis-0.10 + gir1.2-aravis-0.10, or set "
              "ARAVIS_TYPELIB_DIR")
        return False

    try:
        version = Aravis.get_version()
    except Exception:
        version = "unknown"
    print(f"{OK} Aravis {version}")

    if os.environ.get("ARV_FAKE_CAMERA_ENABLED") == "1":
        try:
            Aravis.enable_interface("Fake")
        except Exception:
            pass
    try:
        Aravis.update_device_list()
        n = Aravis.get_n_devices()
        print(f"{OK} {n} camera(s) visible")
        for i in range(n):
            print(f"      [{i:02d}] {Aravis.get_device_id(i)}  "
                  f"{Aravis.get_device_vendor(i)} {Aravis.get_device_model(i)}")
        if n == 0:
            print(f"{WARN} No cameras found — check cabling, MTU and the "
                  "link-local/static IP setup (see README).")
    except Exception as exc:
        print(f"{BAD} Device enumeration failed: {exc}")
        return False
    return True


def check_module(name: str, label: str = "", optional: bool = False) -> bool:
    label = label or name
    try:
        mod = __import__(name)
    except Exception as exc:
        marker = WARN if optional else BAD
        print(f"{marker} {label}: {exc}")
        return False
    version = getattr(mod, "__version__", "")
    print(f"{OK} {label} {version}".rstrip())
    return True


def check_cuda() -> None:
    try:
        import torch
    except Exception:
        print(f"{WARN} PyTorch not installed — CUDA batch mode unavailable "
              "(CPU mode still works)")
        return
    if torch.cuda.is_available():
        print(f"{OK} CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print(f"{WARN} PyTorch present but CUDA is not available — "
              "the app will stay on CPU")


def main() -> int:
    print("── Aravis ──────────────────────────────────────────")
    aravis_ok = check_aravis()

    print("\n── Python dependencies ─────────────────────────────")
    deps_ok = all([
        check_module("PyQt5", "PyQt5"),
        check_module("cv2", "OpenCV"),
        check_module("numpy", "NumPy"),
        check_module("psutil", "psutil"),
        check_module("matplotlib", "matplotlib"),
        check_module("yaml", "PyYAML"),
    ])

    print("\n── GPU (optional) ──────────────────────────────────")
    check_cuda()

    print()
    if aravis_ok and deps_ok:
        print("All required components are present.")
        return 0
    print("Some required components are missing — see the messages above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
