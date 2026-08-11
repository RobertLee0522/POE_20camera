# -*- coding: utf-8 -*-
"""Coverage logging: per-interval averaging, file identity, registry."""

import csv
import os
from datetime import datetime, timedelta

from poe_multi_aravis.services.coverage_logger import (
    CoverageLogger, CoverageLogRegistry,
)


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_header_written_on_creation(tmp_path):
    lg = CoverageLogger("cam-a", str(tmp_path))
    assert os.path.exists(lg.get_filepath())
    assert _rows(lg.get_filepath()) == []


def test_flush_writes_the_mean_of_accumulated_samples(tmp_path):
    lg = CoverageLogger("cam-a", str(tmp_path))
    for v in (10.0, 20.0, 30.0):
        lg.accumulate(v, acquisition_fps=15.0, processing_fps=14.0)
    assert lg.flush() is True

    rows = _rows(lg.get_filepath())
    assert len(rows) == 1
    assert float(rows[0]["coverage_percent"]) == 20.0
    assert rows[0]["samples"] == "3"
    assert rows[0]["camera_id"] == "cam-a"


def test_flush_without_samples_writes_nothing(tmp_path):
    lg = CoverageLogger("cam-a", str(tmp_path))
    assert lg.flush() is False
    assert _rows(lg.get_filepath()) == []


def test_accumulator_resets_between_flushes(tmp_path):
    lg = CoverageLogger("cam-a", str(tmp_path))
    lg.accumulate(100.0)
    lg.flush()
    lg.accumulate(0.0)
    lg.flush()
    rows = _rows(lg.get_filepath())
    assert [float(r["coverage_percent"]) for r in rows] == [100.0, 0.0]


def test_alert_flag_is_sticky_within_an_interval(tmp_path):
    lg = CoverageLogger("cam-a", str(tmp_path))
    lg.accumulate(5.0, alert_active=False)
    lg.accumulate(90.0, alert_active=True)
    lg.accumulate(5.0, alert_active=False)
    lg.flush()
    assert _rows(lg.get_filepath())[0]["alert_active"] == "1"


def test_disabled_logger_records_nothing(tmp_path):
    lg = CoverageLogger("cam-a", str(tmp_path))
    lg.set_enabled(False)
    lg.accumulate(50.0)
    assert lg.flush() is False
    assert _rows(lg.get_filepath()) == []


def test_camera_id_is_sanitised_into_the_filename(tmp_path):
    lg = CoverageLogger("192.168.1.11:cam/1", str(tmp_path))
    assert os.path.basename(lg.get_filepath()) == "192.168.1.11_cam_1.csv"


def test_read_recent_filters_by_age(tmp_path):
    lg = CoverageLogger("cam-a", str(tmp_path))
    lg.accumulate(11.0)
    lg.flush(datetime.now() - timedelta(days=4))
    lg.accumulate(22.0)
    lg.flush(datetime.now())

    recent = lg.read_recent(days=1)
    assert [float(r["coverage_percent"]) for r in recent] == [22.0]
    assert len(lg.read_recent(days=5)) == 2
    assert "_ts" in recent[0]


def test_registry_reuses_one_logger_per_device(tmp_path):
    reg = CoverageLogRegistry(str(tmp_path), interval_seconds=60)
    a1 = reg.get("dev-a", "SN-A")
    a2 = reg.get("dev-a", "SN-A")
    b = reg.get("dev-b", "SN-B")
    assert a1 is a2
    assert b is not a1
    assert reg.find("dev-a") is a1
    assert reg.find("missing") is None


def test_registry_flushes_only_cameras_with_samples(tmp_path):
    reg = CoverageLogRegistry(str(tmp_path))
    reg.get("dev-a", "SN-A").accumulate(10.0)
    reg.get("dev-b", "SN-B")           # no samples
    assert reg.flush_all() == 1


def test_registry_disable_propagates(tmp_path):
    reg = CoverageLogRegistry(str(tmp_path))
    lg = reg.get("dev-a", "SN-A")
    reg.set_enabled(False)
    lg.accumulate(10.0)
    assert reg.flush_all() == 0
