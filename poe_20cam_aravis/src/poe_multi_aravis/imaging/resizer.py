# -*- coding: utf-8 -*-
"""
imaging/resizer.py
==================
Frame resizer – presets, custom width, interpolation control.
Keeps acquisition resolution separate from display/processing resolution.
"""

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np


class FrameResizer:
    """Resizes frames for display/analysis without modifying camera ROI."""

    PRESETS: dict[str, float] = {
        "100%": 1.0,
        "75%":  0.75,
        "50%":  0.5,
        "25%":  0.25,
    }

    INTERPOLATION: dict[str, int] = {
        "linear":  cv2.INTER_LINEAR,
        "cubic":   cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "nearest": cv2.INTER_NEAREST,
    }

    def __init__(self, max_width: Optional[int] = None) -> None:
        self._scale: float = 1.0
        self._custom_w: Optional[int] = None
        self._max_w: Optional[int] = max_width if max_width and max_width > 0 else None
        self._interp: int  = cv2.INTER_LINEAR

    def set_preset(self, name: str) -> bool:
        if name not in self.PRESETS:
            return False
        self._scale    = self.PRESETS[name]
        self._custom_w = None
        return True

    def set_custom_width(self, width: int) -> None:
        self._custom_w = max(10, width)
        self._scale    = 1.0   # overridden by custom_w

    def set_max_width(self, width: Optional[int]) -> None:
        """Cap the display width; wider frames are downscaled, narrow ones left alone.

        This is what keeps 20 simultaneous streams affordable: a 4096-px
        sensor frame is pointless in a tile a few hundred pixels wide, and
        shipping it to the Qt thread costs a full-resolution copy per frame.
        """
        self._max_w = width if width and width > 0 else None

    def set_interpolation(self, method: str) -> bool:
        if method not in self.INTERPOLATION:
            return False
        self._interp = self.INTERPOLATION[method]
        return True

    def resize(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame
        h, w = frame.shape[:2]

        if self._custom_w is not None:
            nw = self._custom_w
            nh = max(1, int(h * nw / w))
        else:
            nw = max(1, int(w * self._scale))
            nh = max(1, int(h * self._scale))

        # Downscale-only cap, applied after the preset/custom width.
        if self._max_w is not None and nw > self._max_w:
            nh = max(1, int(nh * self._max_w / nw))
            nw = self._max_w

        if nw == w and nh == h:
            return frame

        return cv2.resize(frame, (nw, nh), interpolation=self._interp)

    @staticmethod
    def fit_to_widget(frame: np.ndarray, widget_w: int, widget_h: int) -> np.ndarray:
        """Scale frame to fit within widget bounds with letterboxing (aspect-correct)."""
        if frame is None or frame.size == 0 or widget_w <= 0 or widget_h <= 0:
            return frame
        fh, fw = frame.shape[:2]
        scale = min(widget_w / fw, widget_h / fh)
        nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def preset_names() -> list:
        return list(FrameResizer.PRESETS.keys())

    @staticmethod
    def interpolation_names() -> list:
        return list(FrameResizer.INTERPOLATION.keys())
