# -*- coding: utf-8 -*-
"""
services/camera_worker.py
==========================
Per-camera processing worker.

One ``CameraWorker`` QThread exists per connected camera.  It consumes
``Frame`` objects from that camera's ``LatestFrameSlot`` (filled by the
Aravis acquisition thread), applies the display pipeline, and produces
coverage either on the CPU (its own ``CpuCoverageProcessor``) or by
handing the downscaled grayscale to the shared GPU batch processor.

The worker never touches Aravis buffers and never touches Qt widgets —
results leave through signals only.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from ..domain.models import (
    CoverageResult, Frame, GpuBatchItem, ImageStatistics, StreamStatistics,
)
from ..imaging.analyzer import ImageAnalyzer
from ..imaging.background_subtraction import CpuCoverageProcessor
from ..imaging.resizer import FrameResizer
from ..imaging.white_balance import WhiteBalanceController

log = logging.getLogger(__name__)


class CameraWorker(QThread):
    """Frame processing pipeline for exactly one camera."""

    # device_id, display_bgr, CoverageResult | None, ImageStatistics | None
    frame_processed = pyqtSignal(str, np.ndarray, object, object)
    # device_id, acquisition_fps, processing_fps
    fps_updated = pyqtSignal(str, float, float)

    def __init__(
        self,
        device_id: str,
        processor: CpuCoverageProcessor,
        resizer: FrameResizer,
        analyzer: ImageAnalyzer,
        white_balance: WhiteBalanceController,
        target_fps: int = 15,
        threshold: int = 5,
        do_analysis: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.device_id = device_id

        self._processor = processor
        self._resizer   = resizer
        self._analyzer  = analyzer
        self._wb        = white_balance

        self._slot = None                       # LatestFrameSlot
        self._ss: Optional[StreamStatistics] = None
        self._logger = None                     # CoverageLogger
        self._gpu = None                        # GpuBatchProcessor

        self._lock = threading.Lock()
        self._running = True

        self._target_fps  = max(1, target_fps)
        self._threshold   = threshold
        self._do_analysis = do_analysis
        self._do_subtraction = False
        self._proc_mode = "cpu"
        self._capture_bg_pending = False

        # FPS tracking
        self._proc_count = 0
        self._fps_t0 = time.monotonic()
        self._proc_fps = 0.0

    # ── configuration ─────────────────────────────────────────

    def set_source(self, frame_slot, stream_stats: Optional[StreamStatistics]) -> None:
        """Attach the stream's frame slot (or None to idle the worker)."""
        with self._lock:
            self._slot = frame_slot
            self._ss = stream_stats

    def set_logger(self, logger) -> None:
        self._logger = logger

    def set_gpu_processor(self, gpu) -> None:
        """Attach (or detach with None) the shared GPU batch processor."""
        self._gpu = gpu

    def set_processing_mode(self, mode: str) -> None:
        self._proc_mode = "cuda" if mode == "cuda" else "cpu"

    def set_target_fps(self, fps: int) -> None:
        self._target_fps = max(1, fps)

    def set_threshold(self, threshold: int) -> None:
        self._threshold = threshold

    def set_do_analysis(self, enabled: bool) -> None:
        self._do_analysis = enabled

    def set_subtraction(self, enabled: bool) -> None:
        """Enable/disable coverage analysis for this camera.

        Turning it off also drops the background snapshot, so re-enabling
        re-freezes from the next frame — the V4.0.1 behaviour.
        """
        self._do_subtraction = enabled
        if not enabled:
            self.reset_background()

    def get_subtraction(self) -> bool:
        return self._do_subtraction

    def reset_background(self) -> None:
        """Clear the frozen background; the next frame re-freezes it."""
        self._processor.clear_background()
        if self._gpu is not None:
            self._gpu.reset_background(self.device_id)

    def capture_background(self) -> None:
        """Freeze the background from the next processed frame."""
        self.reset_background()
        self._capture_bg_pending = True

    def has_background(self) -> bool:
        if self._proc_mode == "cuda" and self._gpu is not None:
            return self._gpu.has_background(self.device_id)
        return self._processor.has_background()

    def update_alert_threshold(self, pct: float) -> None:
        self._processor.update_alert_threshold(pct)

    # ── QThread.run ───────────────────────────────────────────

    def run(self) -> None:
        while self._running:
            frame_period = 1.0 / self._target_fps
            t0 = time.monotonic()

            with self._lock:
                slot = self._slot
                ss = self._ss

            if slot is None:
                time.sleep(0.05)
                continue

            if not slot.wait(0.05):
                continue

            frame: Optional[Frame] = slot.take()
            if frame is None:
                continue

            try:
                self._process(frame, ss)
            except Exception as exc:
                log.warning("[%s] processing error: %s", self.device_id, exc)

            # FPS limit — the acquisition thread keeps running at the camera's
            # own rate; this only caps how often we pay for the CPU pipeline.
            elapsed = time.monotonic() - t0
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)

        log.debug("[%s] worker exited", self.device_id)

    def stop(self) -> None:
        self._running = False
        self.wait(3000)

    # ── pipeline ──────────────────────────────────────────────

    def _process(self, frame: Frame, ss: Optional[StreamStatistics]) -> None:
        bgr = frame.image

        # Software white balance (skipped entirely at unity gains)
        if (self._wb.r_gain, self._wb.g_gain, self._wb.b_gain) != (1.0, 1.0, 1.0):
            bgr = self._wb.apply(bgr)

        display = self._resizer.resize(bgr)

        stats: Optional[ImageStatistics] = None
        if self._do_analysis:
            try:
                stats = self._analyzer.analyze(display)
            except Exception as exc:
                log.debug("[%s] analysis failed: %s", self.device_id, exc)

        acq_fps = ss.acquisition_fps if ss else 0.0
        self._tick_fps(ss, acq_fps)

        if not self._do_subtraction:
            self.frame_processed.emit(self.device_id, display, None, stats)
            return

        if self._proc_mode == "cuda" and self._gpu is not None:
            aw, ah = self._gpu.analysis_size
            gray_small = cv2.resize(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (aw, ah))
            if self._capture_bg_pending:
                self._capture_bg_pending = False
                self._gpu.reset_background(self.device_id)
            submitted = self._gpu.submit(GpuBatchItem(
                device_id=self.device_id,
                display_bgr=display,
                gray_small=gray_small,
                threshold=self._threshold,
                do_subtraction=True,
                stats=stats,
                acquisition_fps=acq_fps,
                processing_fps=self._proc_fps,
            ))
            if not submitted:
                # GPU is saturated: still show the live frame this round.
                self.frame_processed.emit(self.device_id, display, None, stats)
            return

        # ── CPU path ──
        if self._capture_bg_pending:
            self._capture_bg_pending = False
            self._processor.set_background(bgr)

        coverage: Optional[CoverageResult] = self._processor.process(
            bgr, self._threshold)

        if coverage is not None and self._logger is not None:
            try:
                self._logger.accumulate(
                    coverage_percent=coverage.coverage_percent,
                    acquisition_fps=acq_fps,
                    processing_fps=self._proc_fps,
                    processing_mode=self._proc_mode,
                    alert_active=coverage.alert_active,
                )
            except Exception as exc:
                log.debug("[%s] coverage logging failed: %s", self.device_id, exc)

        self.frame_processed.emit(self.device_id, display, coverage, stats)

    def _tick_fps(self, ss: Optional[StreamStatistics], acq_fps: float) -> None:
        self._proc_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_t0
        if elapsed >= 1.0:
            self._proc_fps = self._proc_count / elapsed
            self._proc_count = 0
            self._fps_t0 = now
            if ss:
                ss.processing_fps = self._proc_fps
            self.fps_updated.emit(self.device_id, acq_fps, self._proc_fps)

    @property
    def processing_fps(self) -> float:
        return self._proc_fps
