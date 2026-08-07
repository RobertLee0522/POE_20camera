# -*- coding: utf-8 -*-
"""
domain/models.py
================
Core data models for the 20-camera POE Aravis application.

Frames and results are immutable frozen dataclasses so they can travel
between the acquisition thread, the per-camera processing worker, the
shared GPU batch processor and the Qt main thread without locking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class CameraDescriptor:
    """Immutable description of a discovered camera."""
    device_id: str
    vendor: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None
    transport: Optional[str] = None
    address: Optional[str] = None          # GigE IP address when known

    def display_name(self) -> str:
        """Short label used in combo boxes and the device list."""
        model = self.model or self.vendor or self.device_id
        if self.address:
            return f"{model}  [{self.address}]"
        if self.serial_number:
            return f"{model}  S/N:{self.serial_number}"
        return model

    def stable_key(self) -> str:
        """Filesystem-safe identity that survives restarts.

        Serial number first, then IP, then the raw device id — so a camera's
        coverage log keeps the same file across sessions even if the
        enumeration order changes.
        """
        raw = self.serial_number or self.address or self.device_id or "camera"
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in raw)

    def __str__(self) -> str:
        return f"{self.display_name()} ({self.device_id})"


@dataclass(frozen=True)
class Frame:
    """
    A single acquired frame with its own NumPy-owned memory.
    The image is always a BGR uint8 ndarray after conversion.
    """
    image: np.ndarray          # BGR uint8, shape (H, W, 3)
    width: int
    height: int
    pixel_format: str
    frame_id: Optional[int]
    camera_timestamp_ns: Optional[int]
    host_timestamp_ns: int = field(default_factory=lambda: time.time_ns())

    def is_color(self) -> bool:
        return self.image.ndim == 3 and self.image.shape[2] == 3

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Frame):
            return NotImplemented
        return (self.frame_id == other.frame_id
                and self.host_timestamp_ns == other.host_timestamp_ns)


@dataclass(frozen=True)
class CameraCapabilities:
    """
    Camera capabilities read at connection time.
    Controls which UI elements are enabled for that camera.
    """
    has_exposure: bool = False
    exposure_min: float = 0.0
    exposure_max: float = 0.0
    exposure_current: float = 0.0

    has_gain: bool = False
    gain_min: float = 0.0
    gain_max: float = 0.0
    gain_current: float = 0.0

    has_frame_rate: bool = False
    fps_min: float = 0.0
    fps_max: float = 0.0
    fps_current: float = 0.0

    has_roi: bool = False
    roi_x: int = 0
    roi_y: int = 0
    roi_w: int = 0
    roi_h: int = 0
    sensor_w: int = 0
    sensor_h: int = 0

    has_trigger: bool = False
    trigger_mode: str = "Off"

    pixel_format: str = ""
    available_pixel_formats: tuple = ()

    vendor: str = ""
    model: str = ""
    serial_number: str = ""
    device_id: str = ""


@dataclass(frozen=True)
class ImageStatistics:
    """Per-frame image analysis results."""
    mean_r: float
    mean_g: float
    mean_b: float
    mean_h: float
    mean_s: float
    mean_l: float
    brightness: float
    width: int
    height: int


@dataclass(frozen=True)
class CoverageResult:
    """Result from the background subtraction + coverage pipeline."""
    coverage_percent: float
    diff_image: np.ndarray        # BGR image with contours drawn
    alert_active: bool
    background_set: bool

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CoverageResult):
            return NotImplemented
        return self.coverage_percent == other.coverage_percent


@dataclass(frozen=True)
class GpuBatchItem:
    """One camera's contribution to a GPU batch.

    ``gray_small`` is already resized to the analysis size so the batch
    processor can stack every camera into a single (N, H, W) upload.
    """
    device_id: str
    display_bgr: np.ndarray
    gray_small: np.ndarray
    threshold: int
    do_subtraction: bool
    stats: Optional[ImageStatistics] = None
    acquisition_fps: float = 0.0
    processing_fps: float = 0.0


@dataclass
class StreamStatistics:
    """Mutable per-camera streaming statistics – updated by the workers."""
    frames_completed: int = 0
    frames_failed: int = 0
    frames_timeout: int = 0
    frames_dropped: int = 0      # dropped because processing was busy
    acquisition_fps: float = 0.0
    display_fps: float = 0.0
    processing_fps: float = 0.0

    def reset(self) -> None:
        self.frames_completed = 0
        self.frames_failed = 0
        self.frames_timeout = 0
        self.frames_dropped = 0
        self.acquisition_fps = 0.0
        self.display_fps = 0.0
        self.processing_fps = 0.0


@dataclass
class CameraRuntimeState:
    """Everything the UI needs to know about one camera at a glance."""
    device_id: str
    index: int
    descriptor: Optional[CameraDescriptor] = None
    capabilities: Optional[CameraCapabilities] = None
    subtraction_enabled: bool = False
    alert_threshold: float = 50.0
    last_coverage: float = 0.0
    connected: bool = False
    streaming: bool = False
