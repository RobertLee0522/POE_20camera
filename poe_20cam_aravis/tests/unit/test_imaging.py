# -*- coding: utf-8 -*-
"""Analyzer, white balance and resizer."""

import numpy as np

from poe_multi_aravis.imaging.analyzer import ImageAnalyzer
from poe_multi_aravis.imaging.resizer import FrameResizer
from poe_multi_aravis.imaging.white_balance import WhiteBalanceController


# ── analyzer ──────────────────────────────────────────────────

def test_analyzer_reports_channel_means_and_size():
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    frame[:, :, 0] = 10      # B
    frame[:, :, 1] = 20      # G
    frame[:, :, 2] = 30      # R
    s = ImageAnalyzer.analyze(frame)
    assert (s.mean_b, s.mean_g, s.mean_r) == (10.0, 20.0, 30.0)
    assert s.width == 80 and s.height == 60


def test_analyzer_brightness_uses_bt601():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[:, :, 2] = 100     # pure red
    s = ImageAnalyzer.analyze(frame)
    assert abs(s.brightness - 29.9) < 0.5


def test_analyzer_handles_empty_frame():
    s = ImageAnalyzer.analyze(np.zeros((0, 0, 3), dtype=np.uint8))
    assert s.width == 0 and s.brightness == 0


# ── white balance ─────────────────────────────────────────────

def test_white_balance_defaults_to_unity():
    wb = WhiteBalanceController()
    assert wb.get_gains() == (1.0, 1.0, 1.0)


def test_white_balance_gains_are_clamped():
    wb = WhiteBalanceController()
    wb.set_gains(9.0, 0.0, 1.5)
    assert wb.get_gains() == (wb.MAX_GAIN, wb.MIN_GAIN, 1.5)


def test_white_balance_apply_scales_channels_and_clips():
    wb = WhiteBalanceController()
    wb.set_gains(2.0, 1.0, 0.5)
    frame = np.full((4, 4, 3), 200, dtype=np.uint8)
    out = wb.apply(frame)
    assert out[0, 0, 2] == 255      # R clipped
    assert out[0, 0, 1] == 200      # G unchanged
    assert out[0, 0, 0] == 100      # B halved
    assert frame[0, 0, 2] == 200    # input untouched


def test_white_balance_presets_and_reset():
    wb = WhiteBalanceController()
    assert wb.set_preset("tungsten") is True
    assert wb.set_preset("no-such-preset") is False
    wb.reset()
    assert wb.get_gains() == (1.0, 1.0, 1.0)


def test_gray_world_pulls_a_colour_cast_back():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, :, 0] = 60      # B
    frame[:, :, 1] = 120     # G
    frame[:, :, 2] = 180     # R
    wb = WhiteBalanceController()
    wb.auto_gray_world(frame)
    assert wb.b_gain > 1.0 > wb.r_gain


# ── resizer ───────────────────────────────────────────────────

def test_resizer_preset_scales():
    r = FrameResizer()
    assert r.set_preset("50%") is True
    out = r.resize(np.zeros((200, 400, 3), dtype=np.uint8))
    assert out.shape[:2] == (100, 200)


def test_resizer_unknown_preset_is_rejected():
    r = FrameResizer()
    assert r.set_preset("33%") is False


def test_resizer_custom_width_keeps_aspect():
    r = FrameResizer()
    r.set_custom_width(100)
    out = r.resize(np.zeros((200, 400, 3), dtype=np.uint8))
    assert out.shape[:2] == (50, 100)


def test_resizer_returns_same_array_at_100_percent():
    r = FrameResizer()
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    assert r.resize(frame) is frame


def test_max_width_downscales_only_wide_frames():
    r = FrameResizer(max_width=640)
    wide = r.resize(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert wide.shape[:2] == (360, 640)

    narrow = np.zeros((240, 320, 3), dtype=np.uint8)
    assert r.resize(narrow) is narrow      # never upscaled


def test_max_width_applies_after_the_preset():
    r = FrameResizer(max_width=400)
    r.set_preset("50%")
    out = r.resize(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert out.shape[1] == 400             # 960 after the preset, then capped


def test_max_width_can_be_disabled():
    r = FrameResizer(max_width=640)
    r.set_max_width(0)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert r.resize(frame) is frame


def test_fit_to_widget_letterboxes():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    out = FrameResizer.fit_to_widget(frame, 100, 100)
    assert out.shape[:2] == (50, 100)
