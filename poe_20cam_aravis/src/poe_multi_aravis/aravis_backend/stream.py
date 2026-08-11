# -*- coding: utf-8 -*-
"""
aravis_backend/stream.py
========================
Aravis stream lifecycle management.

The acquisition worker runs in a dedicated thread.
It pops completed buffers, converts them to Frame objects
with application-owned memory, and immediately requeues
the original Aravis buffer.  The Frame is placed on a
bounded latest-frame exchange (capacity = 1).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Callable

import gi
gi.require_version("Aravis", "0.10")
from gi.repository import Aravis

from ..domain.models import Frame, StreamStatistics
from ..domain.errors import AcquisitionError
from .pixel_formats import PixelConverter, pixel_format_name

log = logging.getLogger(__name__)

_FRAME_TIMEOUT_US = 500_000   # 500 ms default (overridden by settings)


class LatestFrameSlot:
    """
    Thread-safe single-slot frame exchange.
    Older frames are silently replaced (bounded = 1).
    """

    def __init__(self) -> None:
        self._lock  = threading.Lock()
        self._frame: Optional[Frame] = None
        self._event = threading.Event()

    def put(self, frame: Frame) -> None:
        with self._lock:
            self._frame = frame
        self._event.set()

    def take(self) -> Optional[Frame]:
        self._event.clear()
        with self._lock:
            f = self._frame
            self._frame = None
        return f

    def wait(self, timeout: float = 0.1) -> bool:
        return self._event.wait(timeout)

    def clear(self) -> None:
        with self._lock:
            self._frame = None
        self._event.clear()


class AravisStream:
    """
    Manages the complete Aravis stream lifecycle for one camera.

    Lifecycle:
        start() → worker loop → stop()

    The worker thread is started by start() and stopped cleanly by stop().
    No daemon-thread shortcut is used.
    """

    def __init__(
        self,
        camera: Aravis.Camera,
        buffer_count: int = 24,
        frame_timeout_ms: int = 500,
    ) -> None:
        self._camera         = camera
        self._buffer_count   = buffer_count
        self._timeout_us     = frame_timeout_ms * 1_000

        self._stream: Optional[Aravis.Stream] = None
        self._converter   = PixelConverter()
        self._slot        = LatestFrameSlot()
        self._stats       = StreamStatistics()
        self._stop_event  = threading.Event()
        self._worker: Optional[threading.Thread] = None

        # Callbacks
        self._on_frame_error: Optional[Callable[[str], None]] = None
        self._on_consecutive_failure: Optional[Callable[[], None]] = None
        self._consecutive_failures = 0
        self._consecutive_failure_threshold = 5

    # ── public API ────────────────────────────────────────────

    @property
    def latest_frame_slot(self) -> LatestFrameSlot:
        return self._slot

    @property
    def stats(self) -> StreamStatistics:
        return self._stats

    def set_callbacks(
        self,
        on_frame_error: Optional[Callable[[str], None]] = None,
        on_consecutive_failure: Optional[Callable[[], None]] = None,
        consecutive_failure_threshold: int = 5,
    ) -> None:
        self._on_frame_error = on_frame_error
        self._on_consecutive_failure = on_consecutive_failure
        self._consecutive_failure_threshold = consecutive_failure_threshold

    def start(self) -> None:
        """
        Allocate the Aravis stream, fill the buffer pool, and start the
        acquisition worker thread.

        Raises:
            AcquisitionError: if stream creation or camera start fails.
        """
        if self._worker and self._worker.is_alive():
            raise AcquisitionError("Stream is already running")

        self._stats.reset()
        self._stop_event.clear()
        self._slot.clear()
        self._consecutive_failures = 0

        # Create Aravis stream
        try:
            self._stream = self._camera.create_stream(None, None)
        except Exception as exc:
            raise AcquisitionError(f"Failed to create Aravis stream: {exc}") from exc

        if self._stream is None:
            raise AcquisitionError("Aravis camera.create_stream() returned None")

        self._tune_stream(self._stream)

        # Allocate buffer pool
        try:
            payload = self._camera.get_payload()
        except Exception as exc:
            raise AcquisitionError(f"Failed to get payload size: {exc}") from exc

        for _ in range(self._buffer_count):
            self._stream.push_buffer(Aravis.Buffer.new_allocate(payload))

        log.info("Stream started: %d buffers × %d bytes", self._buffer_count, payload)

        # Start camera acquisition
        try:
            self._camera.start_acquisition()
        except Exception as exc:
            self._stream = None
            raise AcquisitionError(f"Camera start_acquisition() failed: {exc}") from exc

        # Launch worker
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="aravis-acquisition",
            daemon=False,
        )
        self._worker.start()

    def stop(self) -> None:
        """
        Signal the acquisition worker to stop, stop camera acquisition,
        wait for the worker, and drain the stream.
        Does NOT disconnect the camera.
        """
        self._stop_event.set()

        try:
            self._camera.stop_acquisition()
        except Exception as exc:
            log.warning("stop_acquisition() raised: %s", exc)

        if self._worker:
            self._worker.join(timeout=5.0)
            if self._worker.is_alive():
                log.warning("Acquisition worker did not exit within 5 s")
            self._worker = None

        self._stream = None
        self._slot.clear()
        log.info("Stream stopped cleanly")

    def is_running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    # ── reliability tuning ────────────────────────────────────

    @staticmethod
    def _tune_stream(stream) -> None:
        """Apply GvStream reliability settings. Never raises.

        A large receive socket buffer plus packet-resend dramatically reduces
        incomplete-buffer drops when streaming high-resolution frames over a
        standard-MTU (non-jumbo) GigE link.
        """
        if not isinstance(stream, Aravis.GvStream):
            return
        settings = [
            ("socket-buffer", Aravis.GvStreamSocketBuffer.AUTO),
            ("packet-resend", Aravis.GvStreamPacketResend.ALWAYS),
            ("packet-timeout", 40_000),      # µs
            ("frame-retention", 200_000),    # µs
        ]
        for name, value in settings:
            try:
                stream.set_property(name, value)
            except Exception as exc:
                log.debug("GvStream property %s not set: %s", name, exc)

    # ── worker ────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        """
        Main acquisition loop.  Runs in a dedicated thread.
        The loop NEVER touches the Qt UI or the processing pipeline.
        """
        _fps_t0       = time.monotonic()
        _fps_count    = 0

        while not self._stop_event.is_set():
            buffer: Optional[Aravis.Buffer] = None
            try:
                buffer = self._stream.timeout_pop_buffer(self._timeout_us)
            except Exception as exc:
                log.debug("timeout_pop_buffer raised: %s", exc)
                self._stats.frames_timeout += 1
                continue

            if buffer is None:
                self._stats.frames_timeout += 1
                log.debug("Buffer timeout")
                continue

            try:
                if buffer.get_status() == Aravis.BufferStatus.SUCCESS:
                    try:
                        frame = self._converter.convert(buffer)
                        # Always return buffer BEFORE publishing frame
                        self._stream.push_buffer(buffer)
                        buffer = None  # prevent double-return in finally

                        # Bounded slot: drop stale if processing is behind
                        self._slot.put(frame)
                        self._stats.frames_completed += 1
                        self._consecutive_failures = 0

                        # FPS tracking
                        _fps_count += 1
                        now = time.monotonic()
                        elapsed = now - _fps_t0
                        if elapsed >= 1.0:
                            self._stats.acquisition_fps = _fps_count / elapsed
                            _fps_count = 0
                            _fps_t0 = now

                    except Exception as exc:
                        if buffer is not None:
                            self._stream.push_buffer(buffer)
                            buffer = None
                        self._stats.frames_failed += 1
                        self._consecutive_failures += 1
                        msg = f"Frame conversion error: {exc}"
                        log.warning(msg)
                        if self._on_frame_error:
                            self._on_frame_error(msg)
                else:
                    status = buffer.get_status()
                    log.debug("Buffer status: %s", status)
                    self._stats.frames_failed += 1
                    self._consecutive_failures += 1
                    if self._on_frame_error:
                        self._on_frame_error(f"Bad buffer status: {status}")

            finally:
                if buffer is not None:
                    try:
                        self._stream.push_buffer(buffer)
                    except Exception:
                        pass

            # Consecutive-failure detection → signal disconnect
            if self._consecutive_failures >= self._consecutive_failure_threshold:
                log.error(
                    "%d consecutive frame failures – signalling disconnect",
                    self._consecutive_failures,
                )
                self._consecutive_failures = 0
                if self._on_consecutive_failure:
                    self._on_consecutive_failure()
                break

        log.debug("Acquisition worker exited")
