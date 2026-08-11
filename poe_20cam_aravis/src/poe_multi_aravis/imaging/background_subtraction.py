# -*- coding: utf-8 -*-
"""
imaging/background_subtraction.py
===================================
Per-camera CPU coverage processor plus the shared drawing helper used by
both the CPU and the GPU (batched) paths.

CPU sequence (identical to the V4.0.1 behaviour of ``multi_cam_stream.py``):
  1. Grayscale conversion
  2. Resize to analysis size (default 640×480)
  3. Gaussian blur
  4. Absolute difference vs the *frozen* background snapshot
  5. Binary threshold
  6. Morphological close + open (ellipse kernel)
  7. External contour detection
  8. Coverage = sum(contour areas) / total pixels × 100
  9. Draw contours + label on the colour diff image

The background is a static snapshot — never a running average — because a
continuously-learning model makes long-running coverage drift towards zero.

With 20 cameras the GPU path does NOT use this class: a single centralised
batch processor (``services/gpu_batch_processor.py``) owns the only CUDA
context and stacks every camera into one upload.  Twenty independent CUDA
contexts would thrash the GPU and the GIL alike.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, runtime_checkable

import cv2
import numpy as np

from ..domain.models import CoverageResult

log = logging.getLogger(__name__)


@runtime_checkable
class CoverageProcessor(Protocol):
    def set_background(self, gray_small: np.ndarray) -> None: ...
    def clear_background(self) -> None: ...
    def process(self, frame: np.ndarray, threshold: int) -> CoverageResult: ...
    def has_background(self) -> bool: ...


# ── shared visualisation ──────────────────────────────────────

def render_diff(mask: np.ndarray, coverage: float) -> np.ndarray:
    """Turn a binary mask into the green-contour diff image shown in the PiP.

    Shared by the CPU and GPU paths so both modes look identical on screen.
    """
    diff_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(diff_bgr, contours, -1, (0, 255, 100), 2)
    cv2.putText(diff_bgr, f"Coverage: {coverage:.1f}%",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2)
    return diff_bgr


def render_placeholder(w: int, h: int, text: str) -> np.ndarray:
    """A dim placeholder tile for 'subtraction off' / 'background not set'."""
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(blank, text, (10, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    return blank


# ── CPU implementation ────────────────────────────────────────

class CpuCoverageProcessor:
    """
    Pure OpenCV/NumPy implementation, one instance per camera.
    Works without any GPU dependency.
    """

    def __init__(
        self,
        analysis_w: int = 640,
        analysis_h: int = 480,
        gaussian_kernel: int = 21,
        morphology_kernel: int = 5,
        alert_threshold: float = 50.0,
    ) -> None:
        self._aw    = analysis_w
        self._ah    = analysis_h
        self._gk    = gaussian_kernel if gaussian_kernel % 2 == 1 else gaussian_kernel + 1
        self._mk    = max(1, morphology_kernel)
        self._alert = alert_threshold

        self._bg: Optional[np.ndarray] = None   # blurred grayscale background
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self._mk, self._mk)
        )

    # ── protocol ──────────────────────────────────────────────

    def set_background(self, frame: np.ndarray) -> None:
        """Freeze the background from a full BGR frame (or a grayscale one)."""
        self._bg = self._prepare(frame).copy()
        log.debug("Background captured: %s", self._bg.shape)

    def clear_background(self) -> None:
        self._bg = None

    def has_background(self) -> bool:
        return self._bg is not None

    def process(self, frame: np.ndarray, threshold: int) -> CoverageResult:
        blurred = self._prepare(frame)

        # First frame after a reset re-freezes the background automatically,
        # which is what the "Reset BG" button relies on.
        if self._bg is None or self._bg.shape != blurred.shape:
            self._bg = blurred.copy()

        diff = cv2.absdiff(self._bg, blurred)
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        closed  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self._morph_kernel)
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  self._morph_kernel)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        total_px   = cleaned.size
        covered_px = sum(cv2.contourArea(c) for c in contours)
        coverage   = (covered_px / total_px * 100.0) if total_px > 0 else 0.0

        return CoverageResult(
            coverage_percent=coverage,
            diff_image=render_diff(cleaned, coverage),
            alert_active=coverage >= self._alert,
            background_set=True,
        )

    # ── tuning ────────────────────────────────────────────────

    def update_alert_threshold(self, pct: float) -> None:
        self._alert = pct

    @property
    def analysis_size(self) -> tuple:
        return (self._aw, self._ah)

    # ── private ───────────────────────────────────────────────

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        """BGR (or gray) frame → blurred grayscale at analysis resolution."""
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        if gray.shape[1] != self._aw or gray.shape[0] != self._ah:
            gray = cv2.resize(gray, (self._aw, self._ah))
        return cv2.GaussianBlur(gray, (self._gk, self._gk), 0)


def make_cpu_processor(
    analysis_w: int = 640,
    analysis_h: int = 480,
    gaussian_kernel: int = 21,
    morphology_kernel: int = 5,
    alert_threshold: float = 50.0,
) -> CpuCoverageProcessor:
    """Factory kept for symmetry with the single-camera application."""
    return CpuCoverageProcessor(
        analysis_w=analysis_w,
        analysis_h=analysis_h,
        gaussian_kernel=gaussian_kernel,
        morphology_kernel=morphology_kernel,
        alert_threshold=alert_threshold,
    )
