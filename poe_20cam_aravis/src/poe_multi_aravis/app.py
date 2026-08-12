# -*- coding: utf-8 -*-
"""
app.py
======
Entry point for the 20-camera POE Aravis monitor.

Usage:
    python -m poe_multi_aravis.app [--config PATH] [--fake N] [--log-level LEVEL]

``--fake N`` enables the Aravis fake-camera interface with N simulated
cameras so the whole application can be exercised without hardware.

Startup order matters and is deliberate:
  1. strip Hikvision MVS paths out of LD_LIBRARY_PATH (re-exec once),
  2. import gi + Aravis *before* PyQt5,
  3. pin Qt's plugin path to PyQt5's own plugins,
  4. only then import Qt and build the window.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Aravis may live under a non-standard prefix (commonly /usr/local when built
# from source).  Override with ARAVIS_TYPELIB_DIR / ARAVIS_PREFIX if needed.
_DEFAULT_TYPELIB_DIRS = [
    os.environ.get("ARAVIS_TYPELIB_DIR", ""),
    "/usr/local/lib/x86_64-linux-gnu/girepository-1.0",
    "/usr/local/lib/girepository-1.0",
    "/usr/lib/x86_64-linux-gnu/girepository-1.0",
    "/usr/lib/aarch64-linux-gnu/girepository-1.0",
]
_DEFAULT_LIB_CANDIDATES = [
    "/usr/local/lib/x86_64-linux-gnu/libaravis-0.10.so.0",
    "/usr/local/lib/libaravis-0.10.so.0",
    "/usr/lib/x86_64-linux-gnu/libaravis-0.10.so.0",
    "/usr/lib/aarch64-linux-gnu/libaravis-0.10.so.0",
    "/usr/local/lib/x86_64-linux-gnu/libaravis-0.8.so.0",
]


def _bootstrap_aravis() -> None:
    """Make ``from gi.repository import Aravis`` work, then do it."""
    existing = os.environ.get("GI_TYPELIB_PATH", "")
    dirs = [d for d in _DEFAULT_TYPELIB_DIRS if d and os.path.isdir(d)]
    if dirs:
        parts = existing.split(os.pathsep) if existing else []
        for d in dirs:
            if d not in parts:
                parts.append(d)
        os.environ["GI_TYPELIB_PATH"] = os.pathsep.join(parts)

    import ctypes
    for cand in _DEFAULT_LIB_CANDIDATES:
        if os.path.exists(cand):
            try:
                ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                continue

    # Import gi + Aravis NOW, before PyQt5 exists in the process.  PyQt5 ships
    # its own GLib; if that loads first, the system libgobject gi needs fails
    # with an undefined-symbol error.
    try:
        import gi
        gi.require_version("Aravis", "0.10")
        from gi.repository import Aravis  # noqa: F401
    except Exception as exc:      # pragma: no cover - surfaced to the user
        raise ImportError(
            "Failed to load the Aravis GObject-Introspection bindings. "
            "Install Aravis 0.10 and its typelib, or set ARAVIS_TYPELIB_DIR / "
            "GI_TYPELIB_PATH. Original error: %s" % exc
        ) from exc


def _sanitize_ld_library_path() -> None:
    """Drop Qt-bundling SDK dirs (e.g. the Hikvision MVS SDK) and re-exec once.

    The old ``multi_cam_stream.py`` needed /opt/MVS on LD_LIBRARY_PATH; if the
    machine still has it exported, its bundled libQt5Core hijacks PyQt5 and the
    application dies at QApplication construction.  The dynamic linker reads
    LD_LIBRARY_PATH only at process start, so a clean value needs a re-exec.
    """
    if os.environ.get("_POE_LD_SANITIZED") == "1":
        return
    ldp = os.environ.get("LD_LIBRARY_PATH", "")
    if not ldp:
        os.environ["_POE_LD_SANITIZED"] = "1"
        return
    bad_markers = ("/opt/MVS", "MVS/lib")
    kept = [p for p in ldp.split(os.pathsep)
            if p and not any(m in p for m in bad_markers)]
    os.environ["_POE_LD_SANITIZED"] = "1"
    new_ldp = os.pathsep.join(kept)
    if new_ldp == ldp:
        return
    if new_ldp:
        os.environ["LD_LIBRARY_PATH"] = new_ldp
    else:
        os.environ.pop("LD_LIBRARY_PATH", None)

    main_mod = sys.modules.get("__main__")
    spec = getattr(main_mod, "__spec__", None)
    if spec is not None and getattr(spec, "name", None):
        os.execv(sys.executable, [sys.executable, "-m", spec.name] + sys.argv[1:])
    else:
        os.execv(sys.executable, [sys.executable] + sys.argv)


def _pin_qt_plugin_path() -> None:
    """Point Qt at PyQt5's own platform plugins.

    Some opencv-python wheels register their bundled Qt plugins globally at
    import time; the wrong 'xcb' plugin then loads and QApplication aborts.
    """
    try:
        import cv2  # noqa: F401  (must be imported before we override)
    except Exception:
        pass
    try:
        import PyQt5
        base = os.path.dirname(PyQt5.__file__)
        for sub in ("Qt5/plugins", "Qt/plugins"):
            plugins = os.path.join(base, sub)
            if os.path.isdir(plugins):
                os.environ["QT_PLUGIN_PATH"] = plugins
                os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
                    plugins, "platforms")
                break
    except Exception:
        pass


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="POE multi-camera (up to 20) Aravis monitor")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--fake", action="store_true",
                        help="Enable the Aravis fake-camera interface. For more "
                             "than one simulated camera, run "
                             "scripts/run_fake_cameras.py first")
    parser.add_argument("--log-level", default="INFO",
                        help="DEBUG / INFO / WARNING / ERROR")
    args = parser.parse_args(argv)

    _sanitize_ld_library_path()   # may re-exec; must precede PyQt5/gi imports
    _configure_logging(args.log_level)

    if args.fake:
        os.environ["ARV_FAKE_CAMERA_ENABLED"] = "1"

    _bootstrap_aravis()
    _pin_qt_plugin_path()

    # Import Qt / Settings AFTER the Aravis bootstrap.
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import QApplication
    from .settings import Settings
    from .ui.main_window import MainWindow
    from .ui.splash import AppSplash
    from .ui import scaling

    # Must be set before the QApplication instance exists.
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv[:1])
    app.setApplicationName("POE Multi-Camera Monitor (Aravis)")

    # Measure the screen the app is opening on and derive the UI scale
    # factor *before* building any widget — every fixed pixel size in the
    # UI reads it at construction time. See ui/scaling.py.
    ui_scale = scaling.init_scale(app)
    logging.getLogger(__name__).info(
        "Screen %sx%s -> UI scale %.2f",
        *scaling.screen_size(), ui_scale)

    # Splash screen: everything below this point (settings validation, main
    # window construction, first device discovery) has no window of its own
    # yet, so give the user visible progress instead of a silent pause.
    splash = AppSplash()
    splash.show()
    splash.step("正在載入設定 / Loading settings…")

    settings = Settings(args.config)

    splash.step("正在建立主視窗 / Building main window…")
    window = MainWindow(settings)

    splash.step("正在偵測相機 / Discovering cameras…")

    _dismissed = {"done": False}

    def _dismiss_splash(*_args) -> None:
        if _dismissed["done"]:
            return
        _dismissed["done"] = True
        splash.finish(window)

    # Closes once the first (auto-triggered) device refresh reports back;
    # a timeout guards against discovery hanging (e.g. a slow/unresponsive
    # GigE link) so the app never gets stuck behind the splash.
    window.service.devices_updated.connect(_dismiss_splash)
    QTimer.singleShot(6000, _dismiss_splash)

    window.showMaximized()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
