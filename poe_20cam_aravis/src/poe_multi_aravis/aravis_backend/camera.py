# -*- coding: utf-8 -*-
"""
aravis_backend/camera.py
=========================
Single-camera session – owns the Aravis.Camera, Feature Adapter,
Stream, and camera state machine.

The CameraSession is the ONLY place that touches Aravis APIs.
UI and services must not call Aravis directly.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Callable

import gi
gi.require_version("Aravis", "0.10")
from gi.repository import Aravis

from ..domain.camera_state import CameraState, CameraStateMachine
from ..domain.errors import (
    CameraConnectionError, AcquisitionError,
    FeatureError, FeatureUnavailableError,
)
from ..domain.models import CameraCapabilities, Frame
from .feature_adapter import FeatureAdapter
from .stream import AravisStream

log = logging.getLogger(__name__)


class CameraSession:
    """
    Manages the complete lifecycle of exactly one camera.

    Thread-safe: methods may be called from any thread, but Aravis
    calls are funnelled through this class.
    """

    def __init__(
        self,
        buffer_count: int = 24,
        frame_timeout_ms: int = 500,
        consecutive_failure_threshold: int = 5,
        on_state_change: Optional[Callable[[CameraState, CameraState], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._buffer_count   = buffer_count
        self._frame_timeout  = frame_timeout_ms
        self._fail_threshold = consecutive_failure_threshold

        self._sm = CameraStateMachine()
        if on_state_change:
            self._sm.add_listener(on_state_change)

        self._on_error = on_error

        self._camera:   Optional[Aravis.Camera]  = None
        self._adapter:  Optional[FeatureAdapter] = None
        self._stream:   Optional[AravisStream]   = None
        self._caps:     Optional[CameraCapabilities] = None

        # Last known device ID (for reconnect)
        self._device_id: Optional[str] = None

        # Last applied camera settings (for reapplying after reconnect)
        self._applied_settings: dict = {}

    # ── state ─────────────────────────────────────────────────

    @property
    def state(self) -> CameraState:
        return self._sm.state

    @property
    def state_machine(self) -> CameraStateMachine:
        return self._sm

    @property
    def capabilities(self) -> Optional[CameraCapabilities]:
        return self._caps

    @property
    def device_id(self) -> Optional[str]:
        return self._device_id

    @property
    def frame_slot(self):
        """The current stream's LatestFrameSlot, or None when not streaming."""
        return self._stream.latest_frame_slot if self._stream else None

    @property
    def stream_stats(self):
        """The current stream's StreamStatistics, or None when not streaming."""
        return self._stream.stats if self._stream else None

    # ── connection ────────────────────────────────────────────

    def connect(self, device_id: str) -> CameraCapabilities:
        """
        Open the camera and read its capabilities.

        Returns:
            CameraCapabilities describing what is available.

        Raises:
            CameraConnectionError on failure.
        """
        try:
            self._sm.transition(CameraState.CONNECTING)
        except ValueError:
            # A raced/failed previous attempt can leave the machine in
            # CONNECTING or another non-connectable state; recover instead of
            # failing the whole reconnect.
            log.warning("connect() from state %s – resetting state machine",
                        self._sm.state.name)
            self._sm.reset()
            self._sm.transition(CameraState.CONNECTING)
        self._device_id = device_id
        log.info("Connecting to %s …", device_id)

        try:
            self._camera = Aravis.Camera.new(device_id)
        except Exception as exc:
            self._sm.force_error()
            raise CameraConnectionError(f"Cannot open camera '{device_id}': {exc}") from exc

        if self._camera is None:
            self._sm.force_error()
            raise CameraConnectionError(f"Aravis.Camera.new('{device_id}') returned None")

        self._adapter = FeatureAdapter(self._camera)
        self._apply_gige_tuning()
        self._caps    = self._read_capabilities(device_id)

        self._sm.transition(CameraState.CONNECTED)
        log.info("Connected: %s", device_id)
        return self._caps

    def disconnect(self) -> None:
        """Stop acquisition (if running) and release camera resources."""
        if self._sm.is_streaming():
            try:
                self.stop_acquisition()
            except Exception:
                pass

        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass
            self._stream = None

        self._camera  = None
        self._adapter = None
        self._caps    = None
        self._sm.reset()
        log.info("Disconnected from %s", self._device_id)

    # ── acquisition ───────────────────────────────────────────

    def start_acquisition(self) -> None:
        if self._camera is None or self._adapter is None:
            raise AcquisitionError("Camera not connected")

        self._sm.transition(CameraState.STARTING)

        self._stream = AravisStream(
            camera=self._camera,
            buffer_count=self._buffer_count,
            frame_timeout_ms=self._frame_timeout,
        )
        self._stream.set_callbacks(
            on_frame_error=self._on_error,
            on_consecutive_failure=self._handle_stream_failure,
            consecutive_failure_threshold=self._fail_threshold,
        )

        try:
            self._stream.start()
        except AcquisitionError:
            self._stream = None
            self._sm.force_error()
            raise

        self._sm.transition(CameraState.STREAMING)
        log.info("Acquisition started")

    def stop_acquisition(self) -> None:
        # Only STREAMING has a valid STOPPING transition; from ERROR (or any
        # other state) we still release the stream but leave the state alone so
        # the caller can drive ERROR → DISCONNECTED / CONNECTING.
        streaming = self._sm.state == CameraState.STREAMING
        if streaming:
            self._sm.transition(CameraState.STOPPING)

        if self._stream:
            try:
                self._stream.stop()
            except Exception as exc:
                log.warning("Stream stop error: %s", exc)
            self._stream = None

        if self._sm.state == CameraState.STOPPING:
            self._sm.transition(CameraState.CONNECTED)
        log.info("Acquisition stopped")

    # ── software trigger ──────────────────────────────────────

    def software_trigger(self) -> None:
        if self._adapter is None:
            return
        try:
            self._adapter.software_trigger()
        except Exception as exc:
            log.warning("Software trigger failed: %s", exc)

    # ── feature access ────────────────────────────────────────

    def read_feature(self, name: str) -> object:
        if self._adapter is None:
            raise FeatureError("Camera not connected")
        return self._adapter.get(name)

    def write_feature(self, name: str, value: object) -> None:
        if self._adapter is None:
            raise FeatureError("Camera not connected")
        self._adapter.set(name, value)

    # ── frames ────────────────────────────────────────────────

    def get_latest_frame(self) -> Optional[Frame]:
        if self._stream is None:
            return None
        return self._stream.latest_frame_slot.take()

    def wait_for_frame(self, timeout: float = 0.1) -> bool:
        if self._stream is None:
            return False
        return self._stream.latest_frame_slot.wait(timeout)

    # ── high-level camera controls ────────────────────────────

    def set_exposure(self, us: float) -> None:
        if self._adapter:
            self._adapter.set_exposure_time(us)
            self._applied_settings["exposure"] = us

    def set_gain(self, db: float) -> None:
        if self._adapter:
            self._adapter.set_gain(db)
            self._applied_settings["gain"] = db

    def set_frame_rate(self, fps: float) -> None:
        if self._adapter:
            self._adapter.set_frame_rate(fps)
            self._applied_settings["frame_rate"] = fps

    def set_region(self, x: int, y: int, w: int, h: int) -> None:
        if self._adapter:
            self._adapter.set_region(x, y, w, h)
            self._applied_settings["region"] = (x, y, w, h)
            self._renegotiate_packet_size()   # payload size changed

    def set_pixel_format(self, fmt: str) -> None:
        if self._adapter:
            self._adapter.set_pixel_format(fmt)
            self._applied_settings["pixel_format"] = fmt
            self._renegotiate_packet_size()   # payload size changed

    def set_trigger_mode(self, enabled: bool) -> None:
        if self._adapter:
            self._adapter.set_trigger_mode(enabled)
            self._applied_settings["trigger"] = enabled

    def auto_white_balance(self) -> bool:
        """Trigger a single hardware auto white balance. Returns success."""
        if self._adapter is None:
            return False
        return self._adapter.auto_white_balance_once()

    def read_current_params(self) -> dict:
        """Read the camera's current exposure/gain/frame-rate/region live."""
        if self._adapter is None:
            return {}
        a = self._adapter

        def _try(fn, default=None):
            try:
                return fn()
            except Exception:
                return default

        x, y, w, h = _try(a.get_region, (0, 0, 0, 0)) or (0, 0, 0, 0)
        return {
            "exposure": _try(a.get_exposure_time, 0.0),
            "gain": _try(a.get_gain, 0.0),
            "frame_rate": _try(a.get_frame_rate, 0.0),
            "pixel_format": _try(a.get_pixel_format, ""),
            "roi_x": x, "roi_y": y, "roi_w": w, "roi_h": h,
        }

    def reset_region_to_max(self) -> tuple[int, int, int, int]:
        """Reset ROI to the full sensor size. Returns (x, y, w, h)."""
        if self._adapter is None:
            return (0, 0, 0, 0)
        sw, sh = self._adapter.get_sensor_size()
        self.set_region(0, 0, int(sw), int(sh))
        return (0, 0, int(sw), int(sh))

    def reapply_settings(self) -> None:
        """Re-apply the last known settings after a reconnect. Never raises."""
        s = dict(self._applied_settings)
        order = ["pixel_format", "region", "exposure", "gain", "frame_rate", "trigger"]
        for key in order:
            if key not in s:
                continue
            try:
                if key == "pixel_format":
                    self._adapter.set_pixel_format(s[key])
                elif key == "region":
                    self._adapter.set_region(*s[key])
                elif key == "exposure":
                    self._adapter.set_exposure_time(s[key])
                elif key == "gain":
                    self._adapter.set_gain(s[key])
                elif key == "frame_rate":
                    self._adapter.set_frame_rate(s[key])
                elif key == "trigger":
                    self._adapter.set_trigger_mode(s[key])
            except Exception as exc:
                log.warning("Reapply %s failed: %s", key, exc)

    # ── private helpers ───────────────────────────────────────

    def _read_capabilities(self, device_id: str) -> CameraCapabilities:
        """Read camera capabilities. Never raises – unknown caps are False."""
        a = self._adapter

        def _try(fn):
            try:
                return fn()
            except Exception:
                return None

        exp = _try(a.get_exposure_time)
        exp_bnds = _try(a.get_exposure_bounds) or (0.0, 0.0)
        gain = _try(a.get_gain)
        gain_bnds = _try(a.get_gain_bounds) or (0.0, 0.0)
        fps = _try(a.get_frame_rate)
        fps_bnds = _try(a.get_frame_rate_bounds) or (0.0, 0.0)
        region = _try(a.get_region)
        sensor = _try(a.get_sensor_size)
        pf = _try(a.get_pixel_format) or ""
        avail_pf = _try(lambda: tuple(self._camera.dup_available_pixel_formats_as_strings())) or ()
        trig = _try(lambda: a.get("TriggerMode")) or "Off"
        sn = _try(lambda: a.get("DeviceSerialNumber")) or ""
        vendor = _try(self._camera.get_vendor_name) or ""
        model  = _try(self._camera.get_model_name) or ""

        return CameraCapabilities(
            has_exposure=exp is not None,
            exposure_min=float(exp_bnds[0]),
            exposure_max=float(exp_bnds[1]),
            exposure_current=float(exp or 0),
            has_gain=gain is not None,
            gain_min=float(gain_bnds[0]),
            gain_max=float(gain_bnds[1]),
            gain_current=float(gain or 0),
            has_frame_rate=fps is not None,
            fps_min=float(fps_bnds[0]),
            fps_max=float(fps_bnds[1]),
            fps_current=float(fps or 0),
            has_roi=region is not None,
            roi_x=int(region[0]) if region else 0,
            roi_y=int(region[1]) if region else 0,
            roi_w=int(region[2]) if region else 0,
            roi_h=int(region[3]) if region else 0,
            sensor_w=int(sensor[0]) if sensor else 0,
            sensor_h=int(sensor[1]) if sensor else 0,
            has_trigger=a.is_available("TriggerMode"),
            trigger_mode=str(trig),
            pixel_format=pf,
            available_pixel_formats=avail_pf,
            serial_number=str(sn),
            vendor=str(vendor),
            model=str(model),
            device_id=device_id,
        )

    def _apply_gige_tuning(self) -> None:
        """Apply GigE-Vision reliability tuning at connect time. Never raises.

        * Force ``GevGVSPExtendedIDMode = Off``.  Some cameras default this to
          ``On`` (64-bit GVSP block/packet IDs), which Aravis 0.10 misparses,
          producing ``PAYLOAD_NOT_SUPPORTED`` buffers and no usable frames.
        * Auto-negotiate the largest working GVSP packet size (jumbo frames
          when the network path supports them).
        """
        cam = self._camera
        try:
            if not cam.is_gv_device():
                return
        except Exception:
            return

        if self._adapter and self._adapter.is_available("GevGVSPExtendedIDMode"):
            try:
                current = str(self._adapter.get("GevGVSPExtendedIDMode"))
            except Exception:
                current = ""
            if current != "Off":
                # Write directly on the device – the node can report itself
                # non-writable through the generic gate on some cameras even
                # when the register accepts the change.
                try:
                    self._camera.get_device().set_string_feature_value(
                        "GevGVSPExtendedIDMode", "Off")
                    log.info("GevGVSPExtendedIDMode set to Off")
                except Exception as exc:
                    log.warning("Could not disable GevGVSPExtendedIDMode "
                                "(camera may misparse GVSP): %s", exc)

        self._renegotiate_packet_size()

    def _renegotiate_packet_size(self) -> None:
        """(Re-)negotiate the GVSP packet size. Safe to call after an ROI or
        pixel-format change (which alter the payload) and at connect time."""
        cam = self._camera
        try:
            if cam is None or not cam.is_gv_device():
                return
            ps = cam.gv_auto_packet_size()
            log.debug("GVSP packet size negotiated: %d bytes", ps)
        except Exception as exc:
            log.debug("gv_auto_packet_size failed: %s", exc)

    def _handle_stream_failure(self) -> None:
        """Called by the stream worker when consecutive failures hit the threshold."""
        log.error("Stream failure threshold reached – entering ERROR state")
        if self._on_error:
            self._on_error("連續幀取得失敗，攝影機可能已斷線 / Consecutive frame failures – camera may be disconnected")
        self._sm.force_error()
