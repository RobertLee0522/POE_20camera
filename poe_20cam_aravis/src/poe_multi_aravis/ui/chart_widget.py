# -*- coding: utf-8 -*-
"""
ui/chart_widget.py
==================
Per-camera coverage history chart (matplotlib embedded in Qt).

Each camera tile owns one of these and can flip between the live view and
the chart.  The chart reads from that camera's CoverageLogger, so it shows
the per-minute averages that survive restarts.
"""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Qt5Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK TC", "Noto Sans CJK SC",
    "Microsoft JhengHei", "Microsoft YaHei", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox

from .i18n import tr
from .scaling import sf, sx
from .theme import C

MAX_DAYS = 5
DEFAULT_DAYS = 1


class ChartWidget(QWidget):
    """Embedded matplotlib chart showing one camera's coverage history."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._days = DEFAULT_DAYS
        self._logger = None          # CoverageLogger

        root = QVBoxLayout(self)
        root.setContentsMargins(sx(4), sx(4), sx(4), sx(4))
        root.setSpacing(sx(4))

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        self._days_lbl = QLabel(tr("chart_days"))
        self._days_lbl.setStyleSheet(
            f"color:{C.TEXT_DIM}; font-size:{sf(12)}px; background:transparent;")
        bar.addWidget(self._days_lbl)

        self._days_combo = QComboBox()
        self._days_combo.setFixedWidth(sx(90))
        for d in range(1, MAX_DAYS + 1):
            self._days_combo.addItem(tr("day_n", d=d), d)
        self._days_combo.setCurrentIndex(DEFAULT_DAYS - 1)
        self._days_combo.currentIndexChanged.connect(self._on_days_changed)
        bar.addWidget(self._days_combo)
        bar.addStretch()
        root.addLayout(bar)

        self._fig = Figure(figsize=(4, 3), dpi=90, facecolor=C.SURFACE)
        self._canvas = FigureCanvas(self._fig)
        self._ax = self._fig.add_subplot(111)
        root.addWidget(self._canvas, 1)

        self._render([], [])

    # ── public API ────────────────────────────────────────────

    def set_logger(self, logger) -> None:
        self._logger = logger
        self.refresh()

    def refresh(self) -> None:
        if self._logger is None:
            self._render([], [])
            return
        rows = self._logger.read_recent(self._days)
        times, values = [], []
        for r in rows:
            times.append(r["_ts"])
            try:
                values.append(float(r["coverage_percent"]))
            except Exception:
                values.append(0.0)
        self._render(times, values)

    def retranslate(self) -> None:
        self._days_lbl.setText(tr("chart_days"))
        for i in range(self._days_combo.count()):
            d = self._days_combo.itemData(i)
            self._days_combo.setItemText(i, tr("day_n", d=d))
        self.refresh()

    # ── private ───────────────────────────────────────────────

    def _on_days_changed(self, _idx: int) -> None:
        self._days = self._days_combo.currentData() or DEFAULT_DAYS
        self.refresh()

    def _render(self, times, values) -> None:
        self._ax.clear()
        self._ax.set_facecolor(C.SURFACE)
        for spine in self._ax.spines.values():
            spine.set_color(C.BORDER)
        self._ax.tick_params(colors="#8899bb", labelsize=8)
        self._ax.set_title(tr("chart_title"), color="#aaccff", fontsize=10)
        self._ax.set_ylabel(tr("chart_ylabel"), color="#88a0cc", fontsize=9)
        self._ax.set_ylim(0, 100)
        self._ax.grid(True, color="#22223a", linewidth=0.5)

        if times:
            self._ax.plot(times, values, color="#55ccee", linewidth=1.4,
                          marker="o", markersize=2.5, markerfacecolor="#aaddff")
            self._ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
            self._fig.autofmt_xdate(rotation=30, ha="right")
        else:
            self._ax.text(0.5, 0.5, tr("chart_no_data"),
                          ha="center", va="center", color="#556699", fontsize=10,
                          transform=self._ax.transAxes)
            self._ax.set_xticks([])

        try:
            self._fig.tight_layout()
        except Exception:
            pass
        self._canvas.draw_idle()
