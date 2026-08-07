# -*- coding: utf-8 -*-
"""CPU coverage processor: frozen background, coverage maths, reset."""

import numpy as np

from poe_multi_aravis.imaging.background_subtraction import (
    CpuCoverageProcessor, render_diff, render_placeholder,
)


def _frame(value=0, w=320, h=240):
    return np.full((h, w, 3), value, dtype=np.uint8)


def _proc(**kw):
    return CpuCoverageProcessor(analysis_w=160, analysis_h=120, **kw)


def test_first_frame_freezes_background_and_reports_zero():
    p = _proc()
    assert not p.has_background()
    result = p.process(_frame(30), threshold=5)
    assert p.has_background()
    assert result.background_set
    assert result.coverage_percent == 0.0


def test_identical_frame_stays_at_zero_coverage():
    p = _proc()
    p.set_background(_frame(60))
    result = p.process(_frame(60), threshold=5)
    assert result.coverage_percent == 0.0
    assert not result.alert_active


def test_full_scene_change_reports_high_coverage():
    p = _proc(alert_threshold=50.0)
    p.set_background(_frame(0))
    result = p.process(_frame(255), threshold=5)
    assert result.coverage_percent > 90.0
    assert result.alert_active


def test_partial_change_is_between_the_extremes():
    p = _proc()
    p.set_background(_frame(0))
    moving = _frame(0)
    moving[:, :160] = 255                      # left half of a 320-wide frame
    result = p.process(moving, threshold=5)
    assert 30.0 < result.coverage_percent < 70.0


def test_threshold_suppresses_small_differences():
    p = _proc()
    p.set_background(_frame(100))
    slight = _frame(103)
    assert p.process(slight, threshold=50).coverage_percent == 0.0
    assert p.process(_frame(255), threshold=50).coverage_percent > 90.0


def test_clear_background_refreezes_on_next_frame():
    p = _proc()
    p.set_background(_frame(0))
    assert p.process(_frame(255), threshold=5).coverage_percent > 90.0
    p.clear_background()
    assert not p.has_background()
    # The re-freeze happens on the next frame, so coverage collapses to zero.
    assert p.process(_frame(255), threshold=5).coverage_percent == 0.0
    assert p.process(_frame(255), threshold=5).coverage_percent == 0.0


def test_resolution_change_refreezes_instead_of_crashing():
    p = _proc()
    p.set_background(_frame(0, w=320, h=240))
    result = p.process(_frame(0, w=640, h=480), threshold=5)
    assert result.background_set


def test_grayscale_input_is_accepted():
    p = _proc()
    gray = np.zeros((240, 320), dtype=np.uint8)
    p.set_background(gray)
    assert p.process(gray, threshold=5).coverage_percent == 0.0


def test_alert_threshold_is_updatable():
    p = _proc(alert_threshold=99.0)
    p.set_background(_frame(0))
    assert not p.process(_frame(255), threshold=5).alert_active
    p.update_alert_threshold(10.0)
    assert p.process(_frame(255), threshold=5).alert_active


def test_diff_image_is_bgr_at_analysis_size():
    p = _proc()
    p.set_background(_frame(0))
    result = p.process(_frame(255), threshold=5)
    assert result.diff_image.shape == (120, 160, 3)
    assert result.diff_image.dtype == np.uint8


def test_render_helpers_produce_bgr_images():
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    assert render_diff(mask, 12.5).shape == (40, 60, 3)
    assert render_placeholder(60, 40, "Subtraction OFF").shape == (40, 60, 3)
