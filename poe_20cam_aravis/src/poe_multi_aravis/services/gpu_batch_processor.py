# -*- coding: utf-8 -*-
"""
services/gpu_batch_processor.py
=================================
Centralised CUDA batch processor for the coverage pipeline.

Why a *single* processor for 20 cameras
---------------------------------------
Giving every camera its own CUDA work would create 20 contexts competing
for the GPU and, worse, 20 Python threads each grabbing the GIL around
tiny kernel launches.  Instead every camera worker hands its downscaled
grayscale frame to this one QThread, which:

  1. drains the queue and forms a batch of up to ``max_batch`` cameras,
  2. stacks them into a single (N, 1, H, W) upload,
  3. runs blur → abs-diff → threshold → morphology → coverage on the whole
     batch at once (each camera keeps its own frozen background tensor and
     its own sensitivity threshold),
  4. downloads only the 1-channel masks and draws the contours on the CPU.

Falls back cleanly: if PyTorch/CUDA is unavailable the processor refuses to
start and the service keeps every camera on the CPU path.
"""

from __future__ import annotations

import ctypes
import logging
import platform
import queue
from typing import Dict, List, Optional

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from ..domain.models import CoverageResult, GpuBatchItem
from ..imaging.background_subtraction import render_diff

log = logging.getLogger(__name__)

_HAS_TORCH = False
try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:      # pragma: no cover - depends on the host install
    pass


def cuda_available() -> bool:
    """True when a usable CUDA device is present."""
    try:
        return _HAS_TORCH and torch.cuda.is_available()
    except Exception:
        return False


def enable_cuda_blocking_sync() -> None:
    """Switch CUDA to blocking-sync so waiting threads sleep instead of spin.

    By default the CUDA runtime busy-waits inside ``.cpu()`` / ``.item()``,
    which pins a core at 100 % per waiting thread — with 20 cameras that made
    the GPU mode burn *more* CPU than the CPU mode.  Must be called before the
    first CUDA context is created.
    """
    try:
        if platform.system() == "Windows":
            candidates = ["cudart64_12.dll", "cudart64_121.dll",
                          "cudart64_120.dll", "cudart64_110.dll",
                          "cudart64_102.dll", "cudart64_101.dll"]
        else:
            candidates = ["libcudart.so", "libcudart.so.12", "libcudart.so.11.0"]
        libcudart = None
        for name in candidates:
            try:
                libcudart = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if libcudart is None:
            return
        cuda_device_schedule_blocking_sync = 0x04
        libcudart.cudaSetDeviceFlags(cuda_device_schedule_blocking_sync)
    except Exception:
        pass


def _make_gaussian_kernel(ksize: int, device):
    """2D Gaussian kernel equivalent to cv2.GaussianBlur(ksize, sigma=0)."""
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8      # OpenCV's default sigma
    ax = torch.arange(ksize, dtype=torch.float32, device=device) - (ksize - 1) / 2.0
    g1d = torch.exp(-(ax ** 2) / (2.0 * sigma * sigma))
    g1d = g1d / g1d.sum()
    return torch.outer(g1d, g1d).reshape(1, 1, ksize, ksize)


class GpuBatchProcessor(QThread):
    """Batched GPU coverage processor shared by every camera."""

    # device_id, display_bgr, CoverageResult, ImageStatistics | None
    frame_processed = pyqtSignal(str, np.ndarray, object, object)

    def __init__(
        self,
        analysis_w: int = 640,
        analysis_h: int = 480,
        gaussian_kernel: int = 21,
        morphology_kernel: int = 5,
        alert_threshold: float = 50.0,
        max_batch: int = 20,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._aw = analysis_w
        self._ah = analysis_h
        self._gk = gaussian_kernel if gaussian_kernel % 2 == 1 else gaussian_kernel + 1
        self._mk = morphology_kernel if morphology_kernel % 2 == 1 else morphology_kernel + 1
        self._alert = alert_threshold
        self._max_batch = max(1, max_batch)

        self._queue: "queue.Queue[Optional[GpuBatchItem]]" = queue.Queue(maxsize=self._max_batch * 4)
        self._running = False
        self._backgrounds: Dict[str, object] = {}   # device_id → GPU tensor

    # ── public API ────────────────────────────────────────────

    @property
    def analysis_size(self) -> tuple:
        return (self._aw, self._ah)

    def submit(self, item: GpuBatchItem) -> bool:
        """Queue one camera's frame. Returns False when the frame was dropped.

        Dropping is intentional: with 20 cameras a blocking put would stall
        the acquisition workers behind the GPU.  A dropped frame is cheaper
        than a stalled pipeline.
        """
        if not self._running:
            return False
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            return False

    def reset_background(self, device_id: str) -> None:
        self._backgrounds.pop(device_id, None)

    def clear_all_backgrounds(self) -> None:
        self._backgrounds.clear()

    def has_background(self, device_id: str) -> bool:
        return device_id in self._backgrounds

    def update_alert_threshold(self, pct: float) -> None:
        self._alert = pct

    def start(self) -> None:                       # type: ignore[override]
        if not cuda_available():
            raise RuntimeError("CUDA is not available")
        try:
            # In GPU mode torch only launches kernels; an intra-op thread pool
            # sized to the CPU core count is pure scheduling overhead here.
            torch.set_num_threads(1)
        except Exception:
            pass
        self._running = True
        self.clear_all_backgrounds()
        super().start()

    def stop(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)           # wake a blocked get()
        except queue.Full:
            pass
        self.wait(5000)
        self.clear_all_backgrounds()

    # ── worker ────────────────────────────────────────────────

    def run(self) -> None:                          # pragma: no cover - needs a GPU
        device = torch.device("cuda")
        gauss = _make_gaussian_kernel(self._gk, device)
        gauss_pad = self._gk // 2
        morph_pad = self._mk // 2

        while self._running:
            batch = self._collect_batch()
            if not batch:
                continue

            # Cameras with subtraction off never touch the GPU.
            sub_items: List[GpuBatchItem] = []
            for item in batch:
                if item.do_subtraction:
                    sub_items.append(item)
                else:
                    self.frame_processed.emit(
                        item.device_id, item.display_bgr, None, item.stats)

            if not sub_items:
                continue

            try:
                self._process_batch(sub_items, device, gauss, gauss_pad, morph_pad)
            except Exception as exc:
                log.warning("GPU batch failed (%d cameras): %s", len(sub_items), exc)
                # Never strand the frames — pass them through un-analysed so the
                # live view keeps updating while the operator switches to CPU.
                for item in sub_items:
                    self.frame_processed.emit(
                        item.device_id, item.display_bgr, None, item.stats)

        log.debug("GPU batch processor exited")

    # ── private ───────────────────────────────────────────────

    def _collect_batch(self) -> List[GpuBatchItem]:
        """Block briefly for one item, then drain whatever else is waiting."""
        try:
            first = self._queue.get(timeout=0.05)
        except queue.Empty:
            return []
        batch: List[GpuBatchItem] = [] if first is None else [first]

        while len(batch) < self._max_batch:
            try:
                nxt = self._queue.get_nowait()
            except queue.Empty:
                break
            if nxt is not None:
                batch.append(nxt)
        return batch

    def _process_batch(self, items, device, gauss, gauss_pad, morph_pad) -> None:
        n = len(items)
        batch_np = np.empty((n, self._ah, self._aw), dtype=np.uint8)
        for i, item in enumerate(items):
            batch_np[i] = item.gray_small

        with torch.inference_mode():
            gray = torch.from_numpy(batch_np).to(
                device, dtype=torch.float32).unsqueeze(1)          # (N,1,H,W)
            gray = F.conv2d(F.pad(gray, (gauss_pad,) * 4, mode="reflect"), gauss)

            # Per-camera frozen background; a missing entry freezes this frame.
            bg_list = []
            for i, item in enumerate(items):
                bg = self._backgrounds.get(item.device_id)
                if bg is None:
                    bg = gray[i:i + 1].clone()
                    self._backgrounds[item.device_id] = bg
                bg_list.append(bg)
            bg_batch = torch.cat(bg_list, dim=0)

            diff = (gray - bg_batch).abs()

            thresholds = torch.tensor(
                [float(it.threshold) for it in items],
                device=device, dtype=torch.float32).view(n, 1, 1, 1)
            binary = (diff > thresholds).to(torch.float32)

            # Morphological close then open, expressed as max/min pools.
            k, p = self._mk, morph_pad
            dilated = F.max_pool2d(binary, kernel_size=k, stride=1, padding=p)
            closed  = 1.0 - F.max_pool2d(1.0 - dilated, kernel_size=k, stride=1, padding=p)
            eroded  = 1.0 - F.max_pool2d(1.0 - closed,  kernel_size=k, stride=1, padding=p)
            cleaned = F.max_pool2d(eroded, kernel_size=k, stride=1, padding=p)

            total_px = self._ah * self._aw
            coverages = (cleaned.sum(dim=(2, 3)).view(n) / total_px * 100.0).cpu().numpy()
            masks = (cleaned.squeeze(1) * 255).to(torch.uint8).cpu().numpy()
            masks = masks.reshape(n, self._ah, self._aw)

        for i, item in enumerate(items):
            cov = float(coverages[i])
            result = CoverageResult(
                coverage_percent=cov,
                diff_image=render_diff(masks[i], cov),
                alert_active=cov >= self._alert,
                background_set=True,
            )
            self.frame_processed.emit(
                item.device_id, item.display_bgr, result, item.stats)
