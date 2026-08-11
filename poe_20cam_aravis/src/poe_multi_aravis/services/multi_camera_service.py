# -*- coding: utf-8 -*-
"""
services/multi_camera_service.py
==================================
Application service layer – the single orchestrator the UI talks to.

It owns:
  * one CameraDiscovery
  * up to ``max_cameras`` CameraSession objects (one Aravis camera each)
  * one CameraWorker QThread per connected camera
  * one shared GpuBatchProcessor (CUDA mode only)
  * one HardwareMonitor
  * one CoverageLogger per camera, flushed on a single timer
  * the shared imaging helpers (analyzer, white balance, resizer)

The UI never touches Aravis, torch, or the imaging modules directly: it
calls methods here and connects to the Qt signals below.  Cameras are
addressed by their Aravis ``device_id`` string throughout.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, List, Optional

import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from ..aravis_backend.camera import CameraSession
from ..aravis_backend.discovery import CameraDiscovery
from ..domain.camera_state import CameraState
from ..domain.errors import PoeError
from ..domain.models import (
    CameraCapabilities, CameraDescriptor, CameraRuntimeState,
)
from ..imaging.analyzer import ImageAnalyzer
from ..imaging.background_subtraction import CpuCoverageProcessor
from ..imaging.resizer import FrameResizer
from ..imaging.white_balance import WhiteBalanceController
from ..settings import Settings
from .camera_worker import CameraWorker
from .coverage_logger import CoverageLogRegistry
from .gpu_batch_processor import (
    GpuBatchProcessor, cuda_available, enable_cuda_blocking_sync,
)
from .hardware_monitor import HardwareMonitor
from .snapshot_service import SnapshotService

log = logging.getLogger(__name__)


class MultiCameraService(QObject):
    """Façade over the whole 20-camera acquisition + processing stack."""

    # ── signals (all delivered on the Qt main thread) ─────────
    devices_updated  = pyqtSignal(list)                  # list[CameraDescriptor]
    state_changed    = pyqtSignal(str, object, object)   # device_id, old, new
    capabilities_ready = pyqtSignal(str, object)         # device_id, CameraCapabilities
    frame_ready      = pyqtSignal(str, np.ndarray, object, object)
    #                  device_id, display_bgr, CoverageResult|None, ImageStatistics|None
    fps_updated      = pyqtSignal(str, float, float)     # device_id, acq, proc
    hardware_stats   = pyqtSignal(float, float, str, str)
    error_message    = pyqtSignal(str)
    info_message     = pyqtSignal(str)
    reconnecting     = pyqtSignal(str, int, int)         # device_id, attempt, max
    reconnected      = pyqtSignal(str)
    logs_flushed     = pyqtSignal()

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self._settings = settings

        self._discovery = CameraDiscovery()
        self._descriptors: List[CameraDescriptor] = []

        # device_id → session / worker / runtime state
        self._sessions: Dict[str, CameraSession] = {}
        self._workers:  Dict[str, CameraWorker]  = {}
        self._runtime:  Dict[str, CameraRuntimeState] = {}

        # shared imaging helpers
        self.analyzer      = ImageAnalyzer()
        self.white_balance = WhiteBalanceController()
        self.resizer       = FrameResizer(max_width=settings.display_max_width)

        # shared GPU processor (created lazily on the first CUDA start)
        self._gpu: Optional[GpuBatchProcessor] = None
        self._proc_mode = settings.processing_mode
        if self._proc_mode == "cuda" and not cuda_available():
            log.warning("processing.mode=cuda but CUDA is unavailable – using CPU")
            self._proc_mode = "cpu"

        self._threshold  = settings.difference_threshold
        self._target_fps = settings.target_fps
        self._alert_threshold = settings.alert_threshold_percent

        # hardware monitor
        self.hardware = HardwareMonitor(settings.monitor_interval_seconds)
        self.hardware.stats_ready.connect(self.hardware_stats)

        # coverage logging
        self.logs = CoverageLogRegistry(
            log_dir=settings.log_directory,
            interval_seconds=settings.log_interval_seconds,
            enabled=settings.logging_enabled,
        )
        self._log_timer = QTimer(self)
        self._log_timer.timeout.connect(self._flush_logs)

        self.snapshots = SnapshotService()

        # reconnect bookkeeping
        self._reconnect_threads: Dict[str, threading.Thread] = {}
        self._reconnect_cancel: Dict[str, threading.Event] = {}
        self._shutting_down = False

    # ── lifecycle ─────────────────────────────────────────────

    def start_background_services(self) -> None:
        self.hardware.start()
        self._log_timer.start(self._settings.log_interval_seconds * 1000)

    def shutdown(self) -> None:
        """Orderly shutdown – no forced thread termination."""
        log.info("MultiCameraService shutting down …")
        self._shutting_down = True
        self._log_timer.stop()

        for ev in self._reconnect_cancel.values():
            ev.set()
        for th in list(self._reconnect_threads.values()):
            if th.is_alive():
                th.join(timeout=2.0)

        self.stop_all()
        self.disconnect_all()

        if self._gpu is not None and self._gpu.isRunning():
            try:
                self._gpu.stop()
            except Exception:
                pass
        try:
            self.hardware.stop()
        except Exception:
            pass

        # One last flush so the final partial interval is not lost.
        try:
            self.logs.flush_all()
        except Exception:
            pass
        log.info("MultiCameraService shut down cleanly")

    # ── discovery ─────────────────────────────────────────────

    def refresh_devices(self) -> List[CameraDescriptor]:
        try:
            devs = self._discovery.refresh()
        except Exception as exc:
            log.error("Discovery failed: %s", exc)
            self.error_message.emit(f"裝置搜尋失敗 / Discovery failed: {exc}")
            devs = []

        devs = devs[: self._settings.max_cameras]
        self._descriptors = devs

        for i, d in enumerate(devs):
            rt = self._runtime.get(d.device_id)
            if rt is None:
                self._runtime[d.device_id] = CameraRuntimeState(
                    device_id=d.device_id,
                    index=i,
                    descriptor=d,
                    alert_threshold=self._alert_threshold,
                )
            else:
                rt.index = i
                rt.descriptor = d

        self.devices_updated.emit(devs)
        return devs

    @property
    def descriptors(self) -> List[CameraDescriptor]:
        return list(self._descriptors)

    def descriptor_for(self, device_id: str) -> Optional[CameraDescriptor]:
        for d in self._descriptors:
            if d.device_id == device_id:
                return d
        return None

    def runtime_for(self, device_id: str) -> Optional[CameraRuntimeState]:
        return self._runtime.get(device_id)

    def connected_ids(self) -> List[str]:
        return list(self._sessions.keys())

    def streaming_count(self) -> int:
        return sum(1 for s in self._sessions.values()
                   if s.state == CameraState.STREAMING)

    # ── connection ────────────────────────────────────────────

    def connect_camera(self, device_id: str) -> bool:
        """Open one camera and spin up its processing worker."""
        if device_id in self._sessions:
            return True

        session = CameraSession(
            buffer_count=self._settings.buffer_count,
            frame_timeout_ms=self._settings.frame_timeout_ms,
            consecutive_failure_threshold=self._settings.consecutive_failure_threshold,
            on_state_change=lambda old, new, d=device_id: self._on_state_change(d, old, new),
            on_error=lambda msg, d=device_id: self._on_backend_error(d, msg),
        )

        try:
            caps: CameraCapabilities = session.connect(device_id)
        except PoeError as exc:
            self.error_message.emit(f"[{self._label(device_id)}] 連線失敗 / Connect failed: {exc}")
            return False
        except Exception as exc:
            self.error_message.emit(f"[{self._label(device_id)}] 連線失敗 / Connect failed: {exc}")
            return False

        self._sessions[device_id] = session
        self._workers[device_id] = self._make_worker(device_id)

        rt = self._runtime.setdefault(
            device_id, CameraRuntimeState(device_id=device_id, index=len(self._runtime)))
        rt.capabilities = caps
        rt.connected = True

        self.capabilities_ready.emit(device_id, caps)
        return True

    def disconnect_camera(self, device_id: str) -> None:
        """Stop the worker, release the camera, and halt its analysis."""
        self._cancel_reconnect(device_id)

        worker = self._workers.pop(device_id, None)
        if worker is not None:
            worker.set_source(None, None)
            worker.set_subtraction(False)
            try:
                worker.stop()
            except Exception:
                pass
            # Only safe once the thread has actually finished — a QThread that
            # is deleted while running takes the process down with it.
            worker.deleteLater()

        session = self._sessions.pop(device_id, None)
        if session is not None:
            try:
                session.disconnect()
            except Exception as exc:
                log.warning("[%s] disconnect error: %s", device_id, exc)

        rt = self._runtime.get(device_id)
        if rt:
            rt.connected = False
            rt.streaming = False
            rt.subtraction_enabled = False

        if self._gpu is not None:
            self._gpu.reset_background(device_id)

    def disconnect_all(self) -> None:
        for device_id in list(self._sessions.keys()):
            self.disconnect_camera(device_id)

    # ── acquisition ───────────────────────────────────────────

    def start_camera(self, device_id: str) -> bool:
        session = self._sessions.get(device_id)
        worker = self._workers.get(device_id)
        if session is None or worker is None:
            return False
        if session.state == CameraState.STREAMING:
            return True                     # already running – "start all" is idempotent
        try:
            session.start_acquisition()
        except Exception as exc:
            # PoeError for a real acquisition failure, ValueError for a raced
            # state transition; neither may abort a 20-camera start-all.
            self.error_message.emit(
                f"[{self._label(device_id)}] 取像啟動失敗 / Start failed: {exc}")
            return False

        worker.set_source(session.frame_slot, session.stream_stats)
        if not worker.isRunning():
            worker.start()

        rt = self._runtime.get(device_id)
        if rt:
            rt.streaming = True
        return True

    def stop_camera(self, device_id: str) -> None:
        worker = self._workers.get(device_id)
        if worker is not None:
            worker.set_source(None, None)
        session = self._sessions.get(device_id)
        if session is not None:
            try:
                session.stop_acquisition()
            except Exception as exc:
                log.warning("[%s] stop error: %s", device_id, exc)
        rt = self._runtime.get(device_id)
        if rt:
            rt.streaming = False

    def start_all(self) -> int:
        """Connect + start every discovered camera. Returns how many started."""
        if self._proc_mode == "cuda":
            self._ensure_gpu_running()

        started = 0
        for d in self._descriptors:
            if not self.connect_camera(d.device_id):
                continue
            if self.start_camera(d.device_id):
                started += 1
        self.info_message.emit(
            f"已啟動 {started} 台相機 / Started {started} camera(s)")
        return started

    def stop_all(self) -> None:
        for device_id in list(self._sessions.keys()):
            self.stop_camera(device_id)
        if self._gpu is not None and self._gpu.isRunning():
            self._gpu.stop()

    # ── processing controls ───────────────────────────────────

    @property
    def processing_mode(self) -> str:
        return self._proc_mode

    def set_processing_mode(self, mode: str) -> str:
        """Switch CPU ↔ CUDA. Returns the mode actually in effect."""
        mode = "cuda" if mode == "cuda" else "cpu"
        if mode == "cuda" and not cuda_available():
            self.info_message.emit(
                "CUDA 不可用，維持 CPU 模式 / CUDA unavailable – staying on CPU")
            mode = "cpu"

        if mode == self._proc_mode:
            return self._proc_mode

        self._proc_mode = mode
        if mode == "cuda":
            self._ensure_gpu_running()
        else:
            if self._gpu is not None and self._gpu.isRunning():
                self._gpu.stop()

        for worker in self._workers.values():
            worker.set_processing_mode(mode)
            worker.set_gpu_processor(self._gpu if mode == "cuda" else None)
            worker.reset_background()
        return self._proc_mode

    def set_subtraction(self, device_id: str, enabled: bool) -> None:
        worker = self._workers.get(device_id)
        if worker is not None:
            worker.set_subtraction(enabled)
        rt = self._runtime.get(device_id)
        if rt:
            rt.subtraction_enabled = enabled
        if enabled:
            # Create the log file up front so the chart has something to open.
            self._logger_for(device_id)

    def set_subtraction_all(self, enabled: bool) -> None:
        for device_id in list(self._workers.keys()):
            self.set_subtraction(device_id, enabled)

    def reset_background(self, device_id: str) -> None:
        worker = self._workers.get(device_id)
        if worker is not None:
            worker.reset_background()

    def reset_all_backgrounds(self) -> None:
        for worker in self._workers.values():
            worker.reset_background()
        if self._gpu is not None:
            self._gpu.clear_all_backgrounds()

    def set_threshold(self, threshold: int) -> None:
        self._threshold = threshold
        for worker in self._workers.values():
            worker.set_threshold(threshold)

    def set_target_fps(self, fps: int) -> None:
        self._target_fps = fps
        for worker in self._workers.values():
            worker.set_target_fps(fps)

    def set_alert_threshold(self, pct: float) -> None:
        self._alert_threshold = pct
        for worker in self._workers.values():
            worker.update_alert_threshold(pct)
        if self._gpu is not None:
            self._gpu.update_alert_threshold(pct)
        for rt in self._runtime.values():
            rt.alert_threshold = pct

    def set_image_analysis(self, enabled: bool) -> None:
        for worker in self._workers.values():
            worker.set_do_analysis(enabled)

    # ── camera parameters ─────────────────────────────────────

    def read_params(self, device_id: str) -> dict:
        session = self._sessions.get(device_id)
        return session.read_current_params() if session else {}

    def apply_params(
        self,
        exposure_us: Optional[float] = None,
        gain_db: Optional[float] = None,
        frame_rate: Optional[float] = None,
        device_ids: Optional[List[str]] = None,
    ) -> tuple:
        """Apply exposure/gain/frame-rate to the given cameras (all by default).

        Returns ``(ok_count, [failure labels])`` – one camera refusing a
        feature must never abort the rest of the fleet.
        """
        targets = device_ids if device_ids is not None else list(self._sessions.keys())
        failures: List[str] = []
        ok = 0
        for device_id in targets:
            session = self._sessions.get(device_id)
            if session is None:
                continue
            errs = []
            for label, value, fn in (
                ("exp",  exposure_us, session.set_exposure),
                ("gain", gain_db,     session.set_gain),
                ("fps",  frame_rate,  session.set_frame_rate),
            ):
                if value is None:
                    continue
                try:
                    fn(value)
                except Exception:
                    errs.append(label)
            if errs:
                failures.append(f"{self._label(device_id)}:{'/'.join(errs)}")
            else:
                ok += 1
        return ok, failures

    def software_trigger(self, device_id: str) -> None:
        session = self._sessions.get(device_id)
        if session is not None:
            session.software_trigger()

    def save_snapshot(self, frame: np.ndarray, device_id: str = "") -> Optional[str]:
        if frame is None:
            self.error_message.emit("尚無影像可儲存 / No frame to save")
            return None
        try:
            prefix = self._safe_key(device_id) if device_id else None
            name = None
            if prefix:
                from datetime import datetime
                name = f"{prefix}_{datetime.now():%Y%m%d_%H%M%S%f}"
            path = self.snapshots.save(frame, filename=name)
            self.info_message.emit(f"已儲存 / Saved: {os.path.basename(path)}")
            return path
        except Exception as exc:
            self.error_message.emit(f"儲存失敗 / Save failed: {exc}")
            return None

    # ── coverage logging ──────────────────────────────────────

    def logger_for(self, device_id: str):
        return self.logs.find(device_id)

    def _logger_for(self, device_id: str):
        """Get (creating on demand) this camera's logger and wire it to its worker.

        Safe to call before the worker exists — the wiring step is skipped and
        ``_make_worker`` picks the logger up itself.
        """
        desc = self.descriptor_for(device_id)
        key = desc.stable_key() if desc else self._safe_key(device_id)
        logger = self.logs.get(device_id, key)
        worker = self._workers.get(device_id)
        if worker is not None:
            worker.set_logger(logger)
        return logger

    def _flush_logs(self) -> None:
        try:
            self.logs.flush_all()
        except Exception as exc:
            log.warning("Coverage log flush failed: %s", exc)
        self.logs_flushed.emit()

    # ── private: worker + GPU wiring ──────────────────────────

    def _make_worker(self, device_id: str) -> CameraWorker:
        processor = CpuCoverageProcessor(
            analysis_w=self._settings.analysis_width,
            analysis_h=self._settings.analysis_height,
            gaussian_kernel=self._settings.gaussian_kernel,
            morphology_kernel=self._settings.morphology_kernel,
            alert_threshold=self._alert_threshold,
        )
        worker = CameraWorker(
            device_id=device_id,
            processor=processor,
            resizer=self.resizer,
            analyzer=self.analyzer,
            white_balance=self.white_balance,
            target_fps=self._target_fps,
            threshold=self._threshold,
            do_analysis=self._settings.enable_image_analysis,
            # Parented to the service so Qt owns the QThread: without an owner
            # Python could collect a worker while its thread is still winding
            # down, which crashes the interpreter.
            parent=self,
        )
        worker.set_processing_mode(self._proc_mode)
        worker.set_gpu_processor(self._gpu if self._proc_mode == "cuda" else None)
        worker.set_logger(self._logger_for(device_id))
        worker.frame_processed.connect(self.frame_ready)
        worker.fps_updated.connect(self.fps_updated)
        return worker

    def _ensure_gpu_running(self) -> None:
        if not cuda_available():
            return
        if self._gpu is None:
            enable_cuda_blocking_sync()
            self._gpu = GpuBatchProcessor(
                analysis_w=self._settings.analysis_width,
                analysis_h=self._settings.analysis_height,
                gaussian_kernel=self._settings.gaussian_kernel,
                morphology_kernel=self._settings.morphology_kernel,
                alert_threshold=self._alert_threshold,
                max_batch=self._settings.gpu_max_batch,
            )
            self._gpu.frame_processed.connect(self._on_gpu_frame)
        if not self._gpu.isRunning():
            try:
                self._gpu.start()
            except Exception as exc:
                log.warning("GPU processor could not start: %s", exc)
                self.info_message.emit(
                    "CUDA 啟動失敗，已回退至 CPU / CUDA start failed – using CPU")
                self._proc_mode = "cpu"
                return
        for worker in self._workers.values():
            worker.set_gpu_processor(self._gpu)
            worker.set_processing_mode("cuda")

    def _on_gpu_frame(self, device_id: str, display_bgr, coverage, stats) -> None:
        """Batched GPU result → coverage log + UI, mirroring the CPU path."""
        if coverage is not None:
            logger = self.logs.find(device_id)
            worker = self._workers.get(device_id)
            if logger is not None:
                try:
                    logger.accumulate(
                        coverage_percent=coverage.coverage_percent,
                        processing_fps=worker.processing_fps if worker else 0.0,
                        processing_mode="cuda",
                        alert_active=coverage.alert_active,
                    )
                except Exception as exc:
                    log.debug("[%s] GPU coverage logging failed: %s", device_id, exc)
        self.frame_ready.emit(device_id, display_bgr, coverage, stats)

    # ── private: backend callbacks + reconnect ────────────────

    def _on_state_change(self, device_id: str, old: CameraState, new: CameraState) -> None:
        rt = self._runtime.get(device_id)
        if rt:
            rt.streaming = (new == CameraState.STREAMING)
            rt.connected = new in (
                CameraState.CONNECTED, CameraState.STARTING,
                CameraState.STREAMING, CameraState.STOPPING,
            )
        self.state_changed.emit(device_id, old, new)

    def _on_backend_error(self, device_id: str, message: str) -> None:
        """Called from a camera's acquisition thread on repeated failures."""
        if self._shutting_down:
            return
        worker = self._workers.get(device_id)
        if worker is not None:
            worker.set_source(None, None)
        self.error_message.emit(f"[{self._label(device_id)}] {message}")
        self._begin_reconnect(device_id)

    def _cancel_reconnect(self, device_id: str) -> None:
        ev = self._reconnect_cancel.get(device_id)
        if ev is not None:
            ev.set()

    def _begin_reconnect(self, device_id: str) -> None:
        if not self._settings.reconnect_enabled or self._shutting_down:
            return
        th = self._reconnect_threads.get(device_id)
        if th is not None and th.is_alive():
            return
        ev = threading.Event()
        self._reconnect_cancel[device_id] = ev
        th = threading.Thread(
            target=self._reconnect_loop,
            args=(device_id, ev),
            name=f"reconnect-{device_id[:16]}",
            daemon=True,
        )
        self._reconnect_threads[device_id] = th
        th.start()

    def _reconnect_loop(self, device_id: str, cancel: threading.Event) -> None:
        max_attempts = self._settings.max_reconnect_attempts
        interval = self._settings.reconnect_interval_seconds

        for attempt in range(1, max_attempts + 1):
            if cancel.wait(interval):
                log.info("[%s] reconnect cancelled", device_id)
                return
            if self._shutting_down:
                return

            self.reconnecting.emit(device_id, attempt, max_attempts)
            session = self._sessions.get(device_id)
            if session is None:
                return
            try:
                try:
                    session.disconnect()
                except Exception:
                    pass
                self._discovery.refresh()
                caps = session.connect(device_id)
                session.reapply_settings()
                self.capabilities_ready.emit(device_id, caps)

                session.start_acquisition()
                worker = self._workers.get(device_id)
                if worker is not None:
                    worker.set_source(session.frame_slot, session.stream_stats)
                self.reconnected.emit(device_id)
                self.info_message.emit(
                    f"[{self._label(device_id)}] 已重新連線 / Reconnected "
                    f"(attempt {attempt})")
                return
            except Exception as exc:
                log.warning("[%s] reconnect %d/%d failed: %s",
                            device_id, attempt, max_attempts, exc)

        self.error_message.emit(
            f"[{self._label(device_id)}] 重新連線失敗，已達最大次數 / "
            f"Reconnect failed after {max_attempts} attempts")

    # ── helpers ───────────────────────────────────────────────

    def _label(self, device_id: str) -> str:
        desc = self.descriptor_for(device_id)
        return desc.display_name() if desc else device_id

    @staticmethod
    def _safe_key(value: str) -> str:
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)
