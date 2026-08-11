# -*- coding: utf-8 -*-
"""
settings.py
===========
Application configuration loader for the 20-camera Aravis monitor.

Reads ``config/default.yaml``, validates every section, and provides typed
defaults for anything missing or out of range.  Never raises on a bad file —
a broken config falls back to the built-in defaults with a warning.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

log = logging.getLogger(__name__)

_DEFAULTS: dict[str, Any] = {
    "camera": {
        "max_cameras": 20,
        # 8 (not 24) by default: 20 cameras × 24 buffers × a multi-megabyte
        # payload would reserve several GB of RAM before a frame arrives.
        "buffer_count": 8,
        "frame_timeout_ms": 500,
        "auto_start_all": False,
        "reconnect_enabled": True,
        "reconnect_interval_seconds": 3,
        "max_reconnect_attempts": 10,
        "consecutive_failure_threshold": 5,
        "packet_delay_ns": 0,
    },
    "processing": {
        "mode": "cpu",                 # "cpu" or "cuda"
        "target_fps": 15,
        "analysis_width": 640,
        "analysis_height": 480,
        "difference_threshold": 5,
        "gaussian_kernel": 21,
        "morphology_kernel": 5,
        "alert_threshold_percent": 50.0,
        "gpu_max_batch": 20,
        "enable_image_analysis": False,
    },
    "logging": {
        "enabled": True,
        "interval_seconds": 60,
        "directory": "./data/coverage",
    },
    "monitoring": {
        "interval_seconds": 1.5,
    },
    "ui": {
        "language": "zh",              # "zh" or "en"
        "cameras_per_page": 2,
        "overview_columns": 4,
        "show_difference_pip": True,
        "display_max_width": 960,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_yaml(path: str) -> dict:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        log.warning("Config file not found: %s – using defaults", path)
        return {}
    except Exception as exc:
        log.warning("Failed to parse config %s: %s – using defaults", path, exc)
        return {}


class Settings:
    """Validated application settings; never raises on missing keys."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        if config_path is None:
            # __file__ = <root>/src/poe_multi_aravis/settings.py → up 2 = <root>
            here = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(here, "..", "..", "config", "default.yaml")

        raw = _load_yaml(os.path.normpath(config_path))
        self._cfg: dict[str, Any] = _deep_merge(_DEFAULTS, raw)
        self._validate()

    def _validate(self) -> None:
        cam = self._cfg["camera"]
        if not (1 <= int(cam["max_cameras"]) <= 64):
            log.warning("max_cameras out of range, resetting to 20")
            cam["max_cameras"] = 20
        if int(cam["buffer_count"]) < 4:
            log.warning("buffer_count too small, resetting to 8")
            cam["buffer_count"] = 8

        proc = self._cfg["processing"]
        if proc["mode"] not in ("cpu", "cuda"):
            log.warning("Invalid processing.mode '%s', using 'cpu'", proc["mode"])
            proc["mode"] = "cpu"
        if not (1 <= int(proc["target_fps"]) <= 120):
            log.warning("target_fps out of range, resetting to 15")
            proc["target_fps"] = 15
        kern = int(proc["gaussian_kernel"])
        if kern % 2 == 0 or kern < 3:
            log.warning("gaussian_kernel must be odd ≥ 3, resetting to 21")
            proc["gaussian_kernel"] = 21
        mkern = int(proc["morphology_kernel"])
        if mkern < 1:
            log.warning("morphology_kernel must be ≥ 1, resetting to 5")
            proc["morphology_kernel"] = 5
        if int(proc["gpu_max_batch"]) < 1:
            proc["gpu_max_batch"] = 20

        ui = self._cfg["ui"]
        if ui["language"] not in ("zh", "en"):
            log.warning("Invalid language '%s', using 'zh'", ui["language"])
            ui["language"] = "zh"
        if not (1 <= int(ui["cameras_per_page"]) <= 8):
            log.warning("cameras_per_page out of range, resetting to 2")
            ui["cameras_per_page"] = 2
        if not (1 <= int(ui["overview_columns"]) <= 10):
            ui["overview_columns"] = 4
        if int(ui["display_max_width"]) < 0:
            ui["display_max_width"] = 960

    # ── camera ────────────────────────────────────────────────

    @property
    def max_cameras(self) -> int:
        return int(self._cfg["camera"]["max_cameras"])

    @property
    def buffer_count(self) -> int:
        return int(self._cfg["camera"]["buffer_count"])

    @property
    def frame_timeout_ms(self) -> int:
        return int(self._cfg["camera"]["frame_timeout_ms"])

    @property
    def auto_start_all(self) -> bool:
        return bool(self._cfg["camera"]["auto_start_all"])

    @property
    def reconnect_enabled(self) -> bool:
        return bool(self._cfg["camera"]["reconnect_enabled"])

    @property
    def reconnect_interval_seconds(self) -> float:
        return float(self._cfg["camera"]["reconnect_interval_seconds"])

    @property
    def max_reconnect_attempts(self) -> int:
        return int(self._cfg["camera"]["max_reconnect_attempts"])

    @property
    def consecutive_failure_threshold(self) -> int:
        return int(self._cfg["camera"]["consecutive_failure_threshold"])

    @property
    def packet_delay_ns(self) -> int:
        return int(self._cfg["camera"]["packet_delay_ns"])

    # ── processing ────────────────────────────────────────────

    @property
    def processing_mode(self) -> str:
        return str(self._cfg["processing"]["mode"])

    @property
    def target_fps(self) -> int:
        return int(self._cfg["processing"]["target_fps"])

    @property
    def analysis_width(self) -> int:
        return int(self._cfg["processing"]["analysis_width"])

    @property
    def analysis_height(self) -> int:
        return int(self._cfg["processing"]["analysis_height"])

    @property
    def difference_threshold(self) -> int:
        return int(self._cfg["processing"]["difference_threshold"])

    @property
    def gaussian_kernel(self) -> int:
        return int(self._cfg["processing"]["gaussian_kernel"])

    @property
    def morphology_kernel(self) -> int:
        return int(self._cfg["processing"]["morphology_kernel"])

    @property
    def alert_threshold_percent(self) -> float:
        return float(self._cfg["processing"]["alert_threshold_percent"])

    @property
    def gpu_max_batch(self) -> int:
        return int(self._cfg["processing"]["gpu_max_batch"])

    @property
    def enable_image_analysis(self) -> bool:
        return bool(self._cfg["processing"]["enable_image_analysis"])

    # ── logging ───────────────────────────────────────────────

    @property
    def logging_enabled(self) -> bool:
        return bool(self._cfg["logging"]["enabled"])

    @property
    def log_interval_seconds(self) -> int:
        return int(self._cfg["logging"]["interval_seconds"])

    @property
    def log_directory(self) -> str:
        return str(self._cfg["logging"]["directory"])

    # ── monitoring ────────────────────────────────────────────

    @property
    def monitor_interval_seconds(self) -> float:
        return float(self._cfg["monitoring"]["interval_seconds"])

    # ── ui ────────────────────────────────────────────────────

    @property
    def language(self) -> str:
        return str(self._cfg["ui"]["language"])

    @language.setter
    def language(self, value: str) -> None:
        if value in ("zh", "en"):
            self._cfg["ui"]["language"] = value

    @property
    def cameras_per_page(self) -> int:
        return int(self._cfg["ui"]["cameras_per_page"])

    @property
    def overview_columns(self) -> int:
        return int(self._cfg["ui"]["overview_columns"])

    @property
    def show_difference_pip(self) -> bool:
        return bool(self._cfg["ui"]["show_difference_pip"])

    @property
    def display_max_width(self) -> int:
        """0 disables the cap and sends frames at acquisition resolution."""
        return int(self._cfg["ui"]["display_max_width"])
