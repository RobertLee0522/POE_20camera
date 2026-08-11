# -*- coding: utf-8 -*-
"""
services/coverage_logger.py
============================
Per-camera CSV coverage logger.

File identity comes from the camera's stable key (serial number, then IP),
so a camera keeps the same log file across restarts even when the Aravis
enumeration order changes.

Samples are *accumulated* by the processing worker and written once per
interval as the mean of that interval — the same behaviour as the V4.0.1
per-minute average, but with the averaging owned by the logger instead of
by a UI widget.  Writes happen off the Qt main thread.

CSV format:
    timestamp, camera_id, coverage_percent, acquisition_fps,
    processing_fps, processing_mode, alert_active, samples
"""

from __future__ import annotations

import csv
import logging
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


class CoverageLogger:
    """Accumulates coverage samples for one camera and flushes the average."""

    FIELDNAMES = [
        "timestamp", "camera_id", "coverage_percent",
        "acquisition_fps", "processing_fps",
        "processing_mode", "alert_active", "samples",
    ]

    def __init__(
        self,
        camera_id: str,
        log_dir: str = "./data/coverage",
        interval_seconds: int = 60,
    ) -> None:
        self._camera_id = camera_id
        self._log_dir   = log_dir
        self._interval  = interval_seconds
        self._enabled   = True

        self._lock = threading.Lock()
        self._cov_sum   = 0.0
        self._acq_sum   = 0.0
        self._proc_sum  = 0.0
        self._count     = 0
        self._alert_any = False
        self._mode      = "cpu"
        self._last_flush: Optional[datetime] = None

        safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in camera_id)
        self._filepath = os.path.join(log_dir, f"{safe_id}.csv")
        self._ensure_file()

    # ── public API ────────────────────────────────────────────

    def accumulate(
        self,
        coverage_percent: float,
        acquisition_fps: float = 0.0,
        processing_fps: float = 0.0,
        processing_mode: str = "cpu",
        alert_active: bool = False,
    ) -> None:
        """Record one sample. Called from the processing worker thread."""
        if not self._enabled:
            return
        with self._lock:
            self._cov_sum  += coverage_percent
            self._acq_sum  += acquisition_fps
            self._proc_sum += processing_fps
            self._count    += 1
            self._alert_any = self._alert_any or alert_active
            self._mode = processing_mode

    def flush(self, when: Optional[datetime] = None) -> bool:
        """Write the accumulated average. Returns True if a row was written."""
        if not self._enabled:
            return False
        with self._lock:
            if self._count == 0:
                return False
            n = self._count
            row = (
                self._cov_sum / n,
                self._acq_sum / n,
                self._proc_sum / n,
                self._mode,
                self._alert_any,
                n,
            )
            self._cov_sum = self._acq_sum = self._proc_sum = 0.0
            self._count = 0
            self._alert_any = False
            self._last_flush = when or datetime.now()

        self._write_row(when or datetime.now(), *row)
        return True

    def try_log(
        self,
        coverage_percent: float,
        acquisition_fps: float = 0.0,
        processing_fps: float = 0.0,
        processing_mode: str = "cpu",
        alert_active: bool = False,
        when: Optional[datetime] = None,
    ) -> bool:
        """Accumulate, then flush if the interval has elapsed.

        Useful when nothing else drives a periodic flush (tests, headless runs).
        """
        self.accumulate(coverage_percent, acquisition_fps, processing_fps,
                        processing_mode, alert_active)
        now = when or datetime.now()
        with self._lock:
            last = self._last_flush
        if last is not None and (now - last).total_seconds() < self._interval:
            return False
        if last is None:
            with self._lock:
                self._last_flush = now
            return False
        return self.flush(now)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def set_interval(self, seconds: int) -> None:
        self._interval = max(10, seconds)

    def get_filepath(self) -> str:
        return self._filepath

    @property
    def camera_id(self) -> str:
        return self._camera_id

    # ── reading for the chart ─────────────────────────────────

    def read_recent(self, days: int) -> List[dict]:
        """Read rows from the last N days, newest last."""
        rows: List[dict] = []
        cutoff = datetime.now() - timedelta(days=days)
        try:
            if not os.path.exists(self._filepath):
                return rows
            with open(self._filepath, "r", newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    try:
                        ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M")
                    except Exception:
                        continue
                    if ts >= cutoff:
                        row["_ts"] = ts
                        rows.append(row)
        except Exception as exc:
            log.warning("Failed to read coverage log %s: %s", self._filepath, exc)
        return rows

    # ── private ───────────────────────────────────────────────

    def _ensure_file(self) -> None:
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            if not os.path.exists(self._filepath):
                with open(self._filepath, "w", newline="", encoding="utf-8") as fh:
                    csv.DictWriter(fh, fieldnames=self.FIELDNAMES).writeheader()
        except Exception as exc:
            log.warning("Cannot create coverage log file %s: %s", self._filepath, exc)

    def _write_row(self, when, coverage, acq, proc, mode, alert, samples) -> None:
        try:
            with open(self._filepath, "a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=self.FIELDNAMES).writerow({
                    "timestamp":        when.strftime("%Y-%m-%d %H:%M"),
                    "camera_id":        self._camera_id,
                    "coverage_percent": f"{coverage:.2f}",
                    "acquisition_fps":  f"{acq:.1f}",
                    "processing_fps":   f"{proc:.1f}",
                    "processing_mode":  mode,
                    "alert_active":     "1" if alert else "0",
                    "samples":          samples,
                })
        except Exception as exc:
            log.warning("Failed to write coverage log row: %s", exc)


class CoverageLogRegistry:
    """Owns one CoverageLogger per camera and flushes them together."""

    def __init__(self, log_dir: str, interval_seconds: int = 60,
                 enabled: bool = True) -> None:
        self._log_dir  = log_dir
        self._interval = interval_seconds
        self._enabled  = enabled
        self._loggers: Dict[str, CoverageLogger] = {}
        self._lock = threading.Lock()

    def get(self, device_id: str, camera_key: str) -> CoverageLogger:
        """Return (creating if needed) the logger for one camera."""
        with self._lock:
            logger = self._loggers.get(device_id)
            if logger is None:
                logger = CoverageLogger(
                    camera_id=camera_key,
                    log_dir=self._log_dir,
                    interval_seconds=self._interval,
                )
                logger.set_enabled(self._enabled)
                self._loggers[device_id] = logger
            return logger

    def find(self, device_id: str) -> Optional[CoverageLogger]:
        with self._lock:
            return self._loggers.get(device_id)

    def flush_all(self, when: Optional[datetime] = None) -> int:
        """Flush every camera's accumulator. Returns the number of rows written."""
        when = when or datetime.now()
        with self._lock:
            loggers = list(self._loggers.values())
        return sum(1 for lg in loggers if lg.flush(when))

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        with self._lock:
            for lg in self._loggers.values():
                lg.set_enabled(enabled)

    @property
    def interval_seconds(self) -> int:
        return self._interval
