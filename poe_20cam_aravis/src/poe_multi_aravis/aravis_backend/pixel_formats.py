# -*- coding: utf-8 -*-
"""
aravis_backend/pixel_formats.py
================================
Pixel-format identification and conversion.  Converts any acquired buffer to a
BGR uint8 numpy array with application-owned memory.

Supported (converted for display/analysis):
  * Mono8 / Mono10 / Mono12 / Mono16          (down-shifted to 8-bit)
  * RGB8Packed / BGR8Packed
  * BayerRG/GR/GB/BG 8, 12, 16                 (demosaiced)
  * YUV422 (YUYV) / YUV422Packed (UYVY)

Unsupported *packed* variants (e.g. BayerRG12Packed, Mono12p) raise a clear
PixelFormatError naming the format so the user can switch to a plain format.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

import gi
gi.require_version("Aravis", "0.10")
from gi.repository import Aravis

from ..domain.models import Frame
from ..domain.errors import PixelFormatError

log = logging.getLogger(__name__)


def _pf(name: str):
    return getattr(Aravis, "PIXEL_FORMAT_" + name, None)


# ── format tables (built defensively; missing constants are skipped) ──

# Mono: fmt_id -> right-shift to reach 8-bit
_MONO = {
    _pf("MONO_8"): 0, _pf("MONO_10"): 2, _pf("MONO_12"): 4, _pf("MONO_16"): 8,
}

# Bayer: fmt_id -> (cv2 demosaic code, right-shift, bytes-per-pixel)
_B = {
    "RG": cv2.COLOR_BAYER_RG2BGR, "GR": cv2.COLOR_BAYER_GR2BGR,
    "GB": cv2.COLOR_BAYER_GB2BGR, "BG": cv2.COLOR_BAYER_BG2BGR,
}
_BAYER: dict = {}
for _p, _code in _B.items():
    for _suffix, _shift, _bpp in (("_8", 0, 1), ("_12", 4, 2), ("_16", 8, 2)):
        _fid = _pf(f"BAYER_{_p}{_suffix}")
        if _fid is not None:
            _BAYER[_fid] = (_code, _shift, _bpp)
    # some cameras expose 10-bit unpacked Bayer
    _fid10 = _pf(f"BAYER_{_p}_10")
    if _fid10 is not None:
        _BAYER[_fid10] = (_code, 2, 2)

# Colour: fmt_id -> cv2 code (None means already BGR)
_COLOR = {}
if _pf("RGB_8_PACKED") is not None:
    _COLOR[_pf("RGB_8_PACKED")] = cv2.COLOR_RGB2BGR
if _pf("BGR_8_PACKED") is not None:
    _COLOR[_pf("BGR_8_PACKED")] = None

# YUV 4:2:2: fmt_id -> cv2 code
_YUV = {}
if _pf("YUV_422_YUYV_PACKED") is not None:
    _YUV[_pf("YUV_422_YUYV_PACKED")] = cv2.COLOR_YUV2BGR_YUYV
if _pf("YUV_422_PACKED") is not None:
    _YUV[_pf("YUV_422_PACKED")] = cv2.COLOR_YUV2BGR_UYVY

# drop any None keys (constants absent in this Aravis build)
for _d in (_MONO, _BAYER, _COLOR, _YUV):
    _d.pop(None, None)

_FORMAT_NAMES = {
    _pf("MONO_8"): "Mono8", _pf("MONO_10"): "Mono10", _pf("MONO_12"): "Mono12",
    _pf("MONO_16"): "Mono16", _pf("RGB_8_PACKED"): "RGB8", _pf("BGR_8_PACKED"): "BGR8",
    _pf("YUV_422_YUYV_PACKED"): "YUV422_YUYV", _pf("YUV_422_PACKED"): "YUV422",
}
for _p in _B:
    for _s in ("_8", "_10", "_12", "_16"):
        _fid = _pf(f"BAYER_{_p}{_s}")
        if _fid is not None:
            _FORMAT_NAMES[_fid] = f"Bayer{_p}{_s[1:]}"
_FORMAT_NAMES.pop(None, None)


def pixel_format_name(fmt_id: int) -> str:
    return _FORMAT_NAMES.get(fmt_id, f"Unknown(0x{fmt_id:08X})")


def is_supported(fmt_id: int) -> bool:
    return fmt_id in _MONO or fmt_id in _BAYER or fmt_id in _COLOR or fmt_id in _YUV


def convert_payload(fmt_id: int, data_bytes: bytes, w: int, h: int) -> np.ndarray:
    """Convert raw payload bytes of a known format to BGR uint8 (owned memory).

    Raises PixelFormatError for unsupported formats.
    """
    # ── Mono ──
    if fmt_id in _MONO:
        shift = _MONO[fmt_id]
        if shift == 0:
            gray = np.frombuffer(data_bytes, dtype=np.uint8).copy().reshape((h, w))
        else:
            u16 = np.frombuffer(data_bytes, dtype=np.uint16).copy().reshape((h, w))
            gray = (u16 >> shift).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ── Bayer ──
    if fmt_id in _BAYER:
        code, shift, bpp = _BAYER[fmt_id]
        if bpp == 1:
            raw8 = np.frombuffer(data_bytes, dtype=np.uint8).copy().reshape((h, w))
        else:
            u16 = np.frombuffer(data_bytes, dtype=np.uint16).copy().reshape((h, w))
            raw8 = (u16 >> shift).astype(np.uint8)
        return cv2.cvtColor(raw8, code)

    # ── RGB / BGR ──
    if fmt_id in _COLOR:
        rgb = np.frombuffer(data_bytes, dtype=np.uint8).copy().reshape((h, w, 3))
        code = _COLOR[fmt_id]
        return rgb if code is None else cv2.cvtColor(rgb, code)

    # ── YUV 4:2:2 (2 bytes/pixel) ──
    if fmt_id in _YUV:
        yuv = np.frombuffer(data_bytes, dtype=np.uint8).copy().reshape((h, w, 2))
        return cv2.cvtColor(yuv, _YUV[fmt_id])

    raise PixelFormatError(
        f"Unsupported pixel format: {pixel_format_name(fmt_id)}. "
        "Switch the camera to Mono8/16, RGB8, BGR8, a plain BayerXX (8/12/16), "
        "or YUV422 — packed variants are not supported."
    )


class PixelConverter:
    """Converts a completed Aravis.Buffer into a Frame with numpy-owned memory.
    The buffer must be returned to the stream immediately after this call."""

    def convert(self, buffer: Aravis.Buffer) -> Frame:
        fmt_id = buffer.get_image_pixel_format()
        w = buffer.get_image_width()
        h = buffer.get_image_height()
        try:
            frame_id = buffer.get_frame_id()
        except Exception:
            frame_id = None
        try:
            cam_ts = buffer.get_timestamp()
        except Exception:
            cam_ts = None

        bgr = convert_payload(fmt_id, buffer.get_data(), w, h)
        return Frame(
            image=bgr, width=w, height=h,
            pixel_format=pixel_format_name(fmt_id),
            frame_id=frame_id, camera_timestamp_ns=cam_ts,
            host_timestamp_ns=time.time_ns(),
        )
