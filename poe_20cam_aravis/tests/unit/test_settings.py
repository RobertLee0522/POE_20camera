# -*- coding: utf-8 -*-
"""Settings loading, merging and validation."""

import os
import textwrap

from poe_multi_aravis.settings import Settings


def _write(tmp_path, body: str) -> str:
    path = tmp_path / "cfg.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(path)


def test_defaults_when_file_missing(tmp_path):
    s = Settings(str(tmp_path / "does-not-exist.yaml"))
    assert s.max_cameras == 20
    assert s.buffer_count == 8
    assert s.processing_mode == "cpu"
    assert s.cameras_per_page == 2
    assert s.language == "zh"


def test_shipped_default_yaml_loads():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    s = Settings(os.path.join(here, "config", "default.yaml"))
    assert s.max_cameras == 20
    assert s.target_fps == 15
    assert s.overview_columns == 4


def test_partial_override_keeps_other_defaults(tmp_path):
    s = Settings(_write(tmp_path, """
        processing:
          target_fps: 25
    """))
    assert s.target_fps == 25
    assert s.difference_threshold == 5      # untouched default
    assert s.max_cameras == 20


def test_invalid_values_fall_back(tmp_path):
    s = Settings(_write(tmp_path, """
        camera:
          max_cameras: 999
          buffer_count: 1
        processing:
          mode: quantum
          target_fps: 0
          gaussian_kernel: 20
          morphology_kernel: 0
        ui:
          language: klingon
          cameras_per_page: 99
    """))
    assert s.max_cameras == 20
    assert s.buffer_count == 8
    assert s.processing_mode == "cpu"
    assert s.target_fps == 15
    assert s.gaussian_kernel == 21
    assert s.morphology_kernel == 5
    assert s.language == "zh"
    assert s.cameras_per_page == 2


def test_malformed_yaml_does_not_raise(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("processing: [unclosed\n", encoding="utf-8")
    s = Settings(str(path))
    assert s.processing_mode == "cpu"


def test_language_setter_rejects_unknown():
    s = Settings("/nonexistent.yaml")
    s.language = "en"
    assert s.language == "en"
    s.language = "fr"
    assert s.language == "en"
