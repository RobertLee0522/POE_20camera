# -*- coding: utf-8 -*-
"""
aravis_backend/feature_adapter.py
==================================
GenICam feature adapter – tries Aravis high-level APIs first,
then generic device feature access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import gi
gi.require_version("Aravis", "0.10")
from gi.repository import Aravis

from ..domain.errors import FeatureError, FeatureUnavailableError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureBounds:
    minimum: Any
    maximum: Any
    increment: Any = None


def _node_kind(node) -> str:
    """Classify an Aravis GenICam node.

    Aravis 0.10 feature nodes do NOT expose ``get_value_type()``; the reliable
    way to determine the value type is ``isinstance`` against the Gc* classes.
    Order matters: GcEnumeration also implements the GcInteger/GcString
    interfaces, and GcCommand must be detected before anything else.
    """
    if isinstance(node, Aravis.GcCommand):
        return "command"
    if isinstance(node, Aravis.GcEnumeration):
        return "enum"          # accessed as a string
    if isinstance(node, Aravis.GcBoolean):
        return "bool"
    if isinstance(node, Aravis.GcFloat):
        return "float"
    if isinstance(node, Aravis.GcInteger):
        return "int"
    if isinstance(node, Aravis.GcString):
        return "str"
    return "str"


class FeatureAdapter:
    """
    Provides a safe, unified interface to camera GenICam features.

    All methods are no-op / return None / return False if the camera
    or device is not set.  Missing features raise FeatureUnavailableError;
    write errors raise FeatureError.
    """

    def __init__(self, camera: Aravis.Camera) -> None:
        self._cam   = camera
        self._dev   = camera.get_device()

    # ── availability / writability ─────────────────────────────

    def is_available(self, feature: str) -> bool:
        try:
            node = self._dev.get_feature(feature)
            return node is not None and node.is_available()
        except Exception:
            return False

    def is_writable(self, feature: str) -> bool:
        try:
            node = self._dev.get_feature(feature)
            return node is not None and node.is_available() and node.is_writable()
        except Exception:
            return False

    # ── bounds ────────────────────────────────────────────────

    def get_bounds(self, feature: str) -> Optional[FeatureBounds]:
        try:
            node = self._dev.get_feature(feature)
            if node is None:
                return None
            kind = _node_kind(node)
            if kind == "int":
                mn, mx = self._dev.get_integer_feature_bounds(feature)
                inc = None
                try:
                    inc = self._dev.get_integer_feature_increment(feature)
                except Exception:
                    pass
                return FeatureBounds(mn, mx, inc)
            if kind == "float":
                mn, mx = self._dev.get_float_feature_bounds(feature)
                return FeatureBounds(mn, mx)
        except Exception:
            pass
        return None

    # ── get ───────────────────────────────────────────────────

    def get(self, feature: str) -> Any:
        if not self.is_available(feature):
            raise FeatureUnavailableError(f"Feature '{feature}' is not available on this camera")
        try:
            node = self._dev.get_feature(feature)
            kind = _node_kind(node)
            if kind == "int":
                return self._dev.get_integer_feature_value(feature)
            if kind == "float":
                return self._dev.get_float_feature_value(feature)
            if kind == "bool":
                return bool(self._dev.get_integer_feature_value(feature))
            # enum / str / command-less → string
            return self._dev.get_string_feature_value(feature)
        except Exception as exc:
            raise FeatureError(f"Failed to read feature '{feature}': {exc}") from exc

    # ── set ───────────────────────────────────────────────────

    def set(self, feature: str, value: Any) -> None:
        if not self.is_available(feature):
            raise FeatureUnavailableError(f"Feature '{feature}' is not available on this camera")
        if not self.is_writable(feature):
            raise FeatureError(f"Feature '{feature}' is read-only")
        try:
            node = self._dev.get_feature(feature)
            kind = _node_kind(node)
            if kind == "int":
                bounds = self.get_bounds(feature)
                v = int(value)
                if bounds:
                    v = max(int(bounds.minimum), min(int(bounds.maximum), v))
                    if bounds.increment:
                        inc = int(bounds.increment)
                        if inc > 1:
                            v = (v // inc) * inc
                self._dev.set_integer_feature_value(feature, v)
            elif kind == "float":
                bounds = self.get_bounds(feature)
                v = float(value)
                if bounds:
                    v = max(float(bounds.minimum), min(float(bounds.maximum), v))
                self._dev.set_float_feature_value(feature, v)
            elif kind == "bool":
                self._dev.set_integer_feature_value(feature, 1 if value else 0)
            else:  # enum / str
                self._dev.set_string_feature_value(feature, str(value))
        except (FeatureUnavailableError, FeatureError):
            raise
        except Exception as exc:
            raise FeatureError(f"Failed to write feature '{feature}'={value!r}: {exc}") from exc

    # ── execute command ────────────────────────────────────────

    def execute(self, command: str) -> None:
        if not self.is_available(command):
            raise FeatureUnavailableError(f"Command '{command}' is not available on this camera")
        try:
            self._dev.execute_command(command)
        except Exception as exc:
            raise FeatureError(f"Failed to execute command '{command}': {exc}") from exc

    # ── high-level convenience wrappers ───────────────────────

    def get_exposure_time(self) -> float:
        try:
            return self._cam.get_exposure_time()
        except Exception:
            return float(self.get("ExposureTime"))

    def set_exposure_time(self, value_us: float) -> None:
        try:
            bounds = self.get_exposure_bounds()
            if bounds:
                value_us = max(bounds[0], min(bounds[1], value_us))
        except Exception:
            pass
        try:
            # Disable auto first
            try:
                self._cam.set_exposure_time_auto(Aravis.Auto.OFF)
            except Exception:
                pass
            self._cam.set_exposure_time(value_us)
        except Exception:
            self.set("ExposureTime", value_us)

    def get_exposure_bounds(self) -> tuple[float, float]:
        try:
            return self._cam.get_exposure_time_bounds()
        except Exception:
            b = self.get_bounds("ExposureTime")
            if b:
                return float(b.minimum), float(b.maximum)
            return 15.0, 9_999_448.0

    def get_gain(self) -> float:
        try:
            return self._cam.get_gain()
        except Exception:
            return float(self.get("Gain"))

    def set_gain(self, value_db: float) -> None:
        try:
            bounds = self.get_gain_bounds()
            if bounds:
                value_db = max(bounds[0], min(bounds[1], value_db))
        except Exception:
            pass
        try:
            try:
                self._cam.set_gain_auto(Aravis.Auto.OFF)
            except Exception:
                pass
            self._cam.set_gain(value_db)
        except Exception:
            self.set("Gain", value_db)

    def get_gain_bounds(self) -> tuple[float, float]:
        try:
            return self._cam.get_gain_bounds()
        except Exception:
            b = self.get_bounds("Gain")
            if b:
                return float(b.minimum), float(b.maximum)
            return 0.0, 24.0

    def get_frame_rate(self) -> float:
        try:
            return self._cam.get_frame_rate()
        except Exception:
            return float(self.get("AcquisitionFrameRate"))

    def set_frame_rate(self, fps: float) -> None:
        try:
            bounds = self.get_frame_rate_bounds()
            if bounds:
                fps = max(bounds[0], min(bounds[1], fps))
        except Exception:
            pass
        try:
            self._cam.set_frame_rate(fps)
        except Exception:
            try:
                self.set("AcquisitionFrameRateEnable", 1)
            except Exception:
                pass
            self.set("AcquisitionFrameRate", fps)

    def get_frame_rate_bounds(self) -> tuple[float, float]:
        try:
            return self._cam.get_frame_rate_bounds()
        except Exception:
            b = self.get_bounds("AcquisitionFrameRate")
            if b:
                return float(b.minimum), float(b.maximum)
            return 0.1, 100.0

    def get_region(self) -> tuple[int, int, int, int]:
        """Returns (x, y, width, height)."""
        try:
            return self._cam.get_region()
        except Exception:
            x = int(self.get("OffsetX") or 0)
            y = int(self.get("OffsetY") or 0)
            w = int(self.get("Width") or 0)
            h = int(self.get("Height") or 0)
            return (x, y, w, h)

    def set_region(self, x: int, y: int, w: int, h: int) -> None:
        try:
            self._cam.set_region(x, y, w, h)
        except Exception:
            self.set("OffsetX", x)
            self.set("OffsetY", y)
            self.set("Width",   w)
            self.set("Height",  h)

    def get_pixel_format(self) -> str:
        try:
            return self._cam.get_pixel_format_as_string()
        except Exception:
            return str(self.get("PixelFormat"))

    def set_pixel_format(self, fmt_str: str) -> None:
        try:
            self._cam.set_pixel_format_from_string(fmt_str)
        except Exception:
            self.set("PixelFormat", fmt_str)

    def software_trigger(self) -> None:
        try:
            self._cam.software_trigger()
        except Exception:
            self.execute("TriggerSoftware")

    def set_trigger_mode(self, enabled: bool) -> None:
        """Enable or disable software trigger mode."""
        try:
            if enabled:
                self._cam.set_trigger("Software")
            else:
                self._cam.clear_triggers()
        except Exception:
            mode = "On" if enabled else "Off"
            try:
                self.set("TriggerSelector", "FrameStart")
            except Exception:
                pass
            self.set("TriggerMode", mode)
            if enabled:
                try:
                    self.set("TriggerSource", "Software")
                except Exception:
                    pass

    def auto_white_balance_once(self) -> bool:
        """Trigger a single hardware auto white balance (BalanceWhiteAuto=Once).
        Returns True if the command was accepted."""
        for value in ("Once", "2"):
            try:
                self.set("BalanceWhiteAuto", value)
                return True
            except Exception:
                continue
        return False

    def has_hardware_white_balance(self) -> bool:
        return self.is_available("BalanceWhiteAuto")

    def get_sensor_size(self) -> tuple[int, int]:
        """Returns (sensor_width, sensor_height)."""
        try:
            return self._cam.get_sensor_size()
        except Exception:
            try:
                # Try reading max W/H from bounds
                wb = self.get_bounds("Width")
                hb = self.get_bounds("Height")
                if wb and hb:
                    return int(wb.maximum), int(hb.maximum)
            except Exception:
                pass
            x, y, w, h = self.get_region()
            return w, h
