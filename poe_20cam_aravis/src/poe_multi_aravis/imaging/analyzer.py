# -*- coding: utf-8 -*-
"""
imaging/analyzer.py
====================
Stateless image analysis – RGB, HSL, brightness statistics.
No Python pixel loops; uses vectorized numpy/OpenCV.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..domain.models import ImageStatistics


class ImageAnalyzer:
    """Analyzes a single BGR uint8 frame."""

    @staticmethod
    def analyze(frame: np.ndarray) -> ImageStatistics:
        """
        Compute RGB/HSL/brightness statistics.

        Args:
            frame: BGR uint8 ndarray.

        Returns:
            ImageStatistics.
        """
        if frame is None or frame.size == 0:
            return ImageStatistics(0, 0, 0, 0, 0, 0, 0, 0, 0)

        h, w = frame.shape[:2]

        # ── RGB means (frame is BGR, so flip channel order) ──
        b = frame[:, :, 0].astype(np.float32)
        g = frame[:, :, 1].astype(np.float32)
        r = frame[:, :, 2].astype(np.float32)
        mean_r = float(r.mean())
        mean_g = float(g.mean())
        mean_b = float(b.mean())

        # ── BT.601 luminance ─────────────────────────────────
        if frame.ndim == 3:
            brightness = float((0.299 * r + 0.587 * g + 0.114 * b).mean())
        else:
            brightness = float(frame.astype(np.float32).mean())

        # ── HSL via OpenCV HLS (order: H, L, S) ──────────────
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        mean_h = float(hls[:, :, 0].mean())          # 0–180
        mean_l = float(hls[:, :, 1].mean() / 255.0 * 100.0)   # %
        mean_s = float(hls[:, :, 2].mean() / 255.0 * 100.0)   # %

        return ImageStatistics(
            mean_r=mean_r,
            mean_g=mean_g,
            mean_b=mean_b,
            mean_h=mean_h,
            mean_s=mean_s,
            mean_l=mean_l,
            brightness=brightness,
            width=w,
            height=h,
        )
