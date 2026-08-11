# -*- coding: utf-8 -*-
"""
services/hardware_monitor.py
=============================
Periodic hardware monitoring worker – CPU, RAM, optional GPU.
Runs in a QThread; emits stats_ready signal to the main thread.
"""

from __future__ import annotations

import subprocess
import time
import platform
import logging

from PyQt5.QtCore import QThread, pyqtSignal
import psutil

log = logging.getLogger(__name__)


class HardwareMonitor(QThread):
    """
    Background thread that periodically samples system metrics.
    All emission is done via Qt signals so the main thread handles display.
    """

    stats_ready = pyqtSignal(float, float, str, str)
    # cpu_pct, ram_pct, gpu_util_str, vram_str

    def __init__(self, interval_seconds: float = 1.5, parent=None) -> None:
        super().__init__(parent)
        self._interval = interval_seconds
        self._running  = True
        self._has_nvidia = True

    def run(self) -> None:
        while self._running:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            gpu_util, vram = "N/A", "N/A"

            if self._has_nvidia:
                try:
                    kwargs: dict = {}
                    if platform.system() == "Windows":
                        import subprocess
                        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                    result = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits",
                        ],
                        timeout=2,
                        **kwargs,
                    ).decode().strip()
                    parts = result.split(",")
                    if len(parts) == 3:
                        u, mu, mt = parts
                        gpu_util = f"{u.strip()}%"
                        vram     = f"{mu.strip()} / {mt.strip()} MB"
                except FileNotFoundError:
                    self._has_nvidia = False
                    gpu_util = "N/A"
                    vram     = "N/A"
                except Exception:
                    gpu_util = "N/A"
                    vram     = "N/A"

            self.stats_ready.emit(cpu, ram, gpu_util, vram)
            time.sleep(self._interval)

    def stop(self) -> None:
        self._running = False
        self.wait(3000)
