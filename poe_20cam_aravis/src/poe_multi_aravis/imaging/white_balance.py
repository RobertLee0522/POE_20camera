# -*- coding: utf-8 -*-
"""
imaging/white_balance.py
=========================
Software white balance – manual gains, 5 presets, gray-world auto.
Hardware white-balance is handled separately via the FeatureAdapter.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


class WhiteBalanceController:
    """Software white-balance applied to display/analysis frames."""

    PRESETS: dict[str, Tuple[float, float, float]] = {
        "default":     (1.0,  1.0,  1.0),
        "daylight":    (1.0,  1.0,  1.1),
        "cloudy":      (1.15, 1.0,  1.0),
        "tungsten":    (1.5,  1.0,  0.8),
        "fluorescent": (0.9,  1.0,  1.2),
        "shade":       (1.2,  1.0,  0.9),
    }

    MIN_GAIN = 0.5
    MAX_GAIN = 2.0

    def __init__(self) -> None:
        self.r_gain = 1.0
        self.g_gain = 1.0
        self.b_gain = 1.0

    # ── setters ───────────────────────────────────────────────

    def set_gains(self, r: float, g: float, b: float) -> None:
        self.r_gain = self._clamp(r)
        self.g_gain = self._clamp(g)
        self.b_gain = self._clamp(b)

    def set_preset(self, name: str) -> bool:
        if name not in self.PRESETS:
            return False
        self.set_gains(*self.PRESETS[name])
        return True

    def reset(self) -> None:
        self.r_gain = 1.0
        self.g_gain = 1.0
        self.b_gain = 1.0

    def auto_gray_world(self, frame: np.ndarray) -> None:
        """Gray-world automatic white balance."""
        if frame is None or frame.size == 0:
            return
        b_mean = float(frame[:, :, 0].mean())
        g_mean = float(frame[:, :, 1].mean())
        r_mean = float(frame[:, :, 2].mean())
        global_mean = (b_mean + g_mean + r_mean) / 3.0
        if r_mean > 0:
            self.r_gain = self._clamp(global_mean / r_mean)
        if g_mean > 0:
            self.g_gain = self._clamp(global_mean / g_mean)
        if b_mean > 0:
            self.b_gain = self._clamp(global_mean / b_mean)

    # ── apply ─────────────────────────────────────────────────

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply current gains to a BGR uint8 frame. Returns a new array."""
        if frame is None or frame.size == 0:
            return frame
        adj = frame.astype(np.float32)
        adj[:, :, 0] *= self.b_gain
        adj[:, :, 1] *= self.g_gain
        adj[:, :, 2] *= self.r_gain
        return np.clip(adj, 0, 255).astype(np.uint8)

    # ── getters ───────────────────────────────────────────────

    def get_gains(self) -> Tuple[float, float, float]:
        return (self.r_gain, self.g_gain, self.b_gain)

    @staticmethod
    def preset_names() -> list:
        return list(WhiteBalanceController.PRESETS.keys())

    # ── helpers ───────────────────────────────────────────────

    def _clamp(self, v: float) -> float:
        return max(self.MIN_GAIN, min(self.MAX_GAIN, v))
