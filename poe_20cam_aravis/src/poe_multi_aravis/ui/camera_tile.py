# -*- coding: utf-8 -*-
"""
ui/camera_tile.py
=================
One full-size camera slot on a tab page.

    ┌── toolbar ────────────────────────────────────────────────────┐
    │ #01 [camera ▾] 1920×1080 15.0f ▣ 0.0% ⚠[50%] [📈][BG][Sub] ● │
    ├───────────────────────────────────────────────────────────────┤
    │  live video (double-click → fullscreen)                       │
    │                                     ┌── PiP diff ──┐          │
    └───────────────────────────────────────────────────────────────┘

A stacked widget flips the video area to the coverage chart.  The frame
border flashes red at 400 ms while a coverage alert is active.
"""

from __future__ import annotations

import time
from typing import List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QStackedWidget, QVBoxLayout, QWidget,
)

from ..domain.models import CameraDescriptor
from .chart_widget import ChartWidget
from .fullscreen_dialog import FullscreenDialog
from .i18n import tr
from .theme import C, btn_ghost, btn_toggle

_STYLE_NORMAL = f"""
    CameraTile {{
        border: 1px solid {C.BORDER};
        border-radius: 12px;
        background: {C.SURFACE};
    }}
"""
_STYLE_FLASH_ON = """
    CameraTile {
        border: 3px solid #ff2b2b;
        border-radius: 12px;
        background: #1e0606;
    }
"""
_STYLE_FLASH_OFF = """
    CameraTile {
        border: 3px solid #5c1010;
        border-radius: 12px;
        background: #170404;
    }
"""


class CameraTile(QFrame):
    """Full camera slot with live view, PiP diff, chart and per-camera controls."""

    subtraction_toggled = pyqtSignal(str, bool)   # device_id, enabled
    slot_device_changed = pyqtSignal(int, str)    # slot_id, device_id ("" = unassigned)
    reset_bg_requested  = pyqtSignal(str)         # device_id

    def __init__(self, slot_id: int, descriptors: List[CameraDescriptor],
                 alert_threshold: float = 50.0, show_pip: bool = True,
                 parent=None) -> None:
        super().__init__(parent)
        self.slot_id = slot_id
        self._descriptors = list(descriptors)
        self._device_id = ""
        self._sub_enabled = False
        self._alert_threshold = alert_threshold
        self._alert_active = False
        self._flash_state = False
        self._chart_mode = False
        self._show_pip = show_pip
        self._conn_state = "waiting"      # waiting / streaming / stopped
        self._last_bgr: Optional[np.ndarray] = None
        self._last_frame_time = 0.0
        self._fullscreen: Optional[FullscreenDialog] = None

        self._flash_timer = QTimer(self)
        self._flash_timer.setInterval(400)
        self._flash_timer.timeout.connect(self._on_flash_tick)

        self._build_ui()
        self.set_descriptors(self._descriptors, preferred_index=slot_id)

    # ── construction ──────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setFrameShape(QFrame.Box)
        self.setStyleSheet(_STYLE_NORMAL)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(6)

        bar = QHBoxLayout()
        bar.setSpacing(5)
        bar.setContentsMargins(0, 0, 0, 0)

        slot_lbl = QLabel(f"#{self.slot_id + 1:02d}")
        slot_lbl.setFixedWidth(32)
        slot_lbl.setStyleSheet(
            f"color:{C.ACCENT}; font-weight:700; font-size:14px; background:transparent;")
        bar.addWidget(slot_lbl)

        self.cam_combo = QComboBox()
        self.cam_combo.setMinimumHeight(30)
        self.cam_combo.setMinimumWidth(120)
        self.cam_combo.currentIndexChanged.connect(self._on_combo_changed)
        bar.addWidget(self.cam_combo, 1)

        self.res_lbl = QLabel("----×----")
        self.res_lbl.setFixedWidth(72)
        self.res_lbl.setStyleSheet(
            f"color:{C.TEXT_FAINT}; font-size:12px; background:transparent;"
            " qproperty-alignment:AlignRight;")
        bar.addWidget(self.res_lbl)

        self.fps_lbl = QLabel("--")
        self.fps_lbl.setFixedWidth(58)
        self.fps_lbl.setStyleSheet(
            f"color:{C.WARNING}; font-size:12px; font-weight:600;"
            " background:transparent; qproperty-alignment:AlignRight;")
        bar.addWidget(self.fps_lbl)

        cov_icon = QLabel("▣")
        cov_icon.setStyleSheet(f"color:{C.ACCENT}; font-size:14px; background:transparent;")
        bar.addWidget(cov_icon)
        self.cov_lbl = QLabel("--")
        self.cov_lbl.setFixedWidth(54)
        self.cov_lbl.setStyleSheet(
            f"color:{C.ACCENT_2}; font-size:13px; font-weight:700; background:transparent;")
        bar.addWidget(self.cov_lbl)

        alert_icon = QLabel("⚠")
        alert_icon.setStyleSheet(f"color:{C.WARNING}; font-size:14px; background:transparent;")
        bar.addWidget(alert_icon)
        self.alert_spin = QDoubleSpinBox()
        self.alert_spin.setRange(0.0, 100.0)
        self.alert_spin.setValue(self._alert_threshold)
        self.alert_spin.setSuffix("%")
        self.alert_spin.setDecimals(0)
        self.alert_spin.setFixedWidth(74)
        self.alert_spin.setMinimumHeight(30)
        self.alert_spin.setToolTip(tr("alert_spin_tip"))
        self.alert_spin.valueChanged.connect(self._on_alert_threshold_changed)
        bar.addWidget(self.alert_spin)

        self.alert_lbl = QLabel()
        self.alert_lbl.setFixedWidth(52)
        self.alert_lbl.setStyleSheet(
            f"color:{C.DANGER}; font-size:12px; font-weight:700; background:transparent;")
        bar.addWidget(self.alert_lbl)

        self.view_btn = QPushButton(tr("show_chart"))
        self.view_btn.setFixedHeight(30)
        self.view_btn.setStyleSheet(btn_ghost())
        self.view_btn.clicked.connect(self._toggle_view)
        bar.addWidget(self.view_btn)

        self.reset_bg_btn = QPushButton(tr("reset_bg"))
        self.reset_bg_btn.setFixedHeight(30)
        self.reset_bg_btn.setToolTip(tr("reset_bg_tip"))
        self.reset_bg_btn.setStyleSheet(btn_ghost())
        self.reset_bg_btn.clicked.connect(
            lambda: self.reset_bg_requested.emit(self._device_id))
        bar.addWidget(self.reset_bg_btn)

        self.sub_btn = QPushButton(tr("subtract"))
        self.sub_btn.setCheckable(True)
        self.sub_btn.setFixedHeight(30)
        self.sub_btn.setMinimumWidth(80)
        self.sub_btn.setStyleSheet(btn_toggle())
        self.sub_btn.clicked.connect(self._on_sub_toggled)
        bar.addWidget(self.sub_btn)

        self.status_dot = QLabel("●")
        self.status_dot.setFixedWidth(18)
        self.status_dot.setStyleSheet(
            f"color:{C.TEXT_FAINT}; font-size:16px; background:transparent;")
        self.status_dot.setToolTip(tr("status_disconnected"))
        bar.addWidget(self.status_dot)

        root.addLayout(bar)

        # ── video area ──
        self.img_container = QWidget()
        self.img_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img_container.setMinimumHeight(140)
        self.img_container.setStyleSheet("background:#05070c; border-radius:8px;")

        self.video_lbl = QLabel(self.img_container)
        self.video_lbl.setAlignment(Qt.AlignCenter)
        self.video_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_lbl.setStyleSheet(
            f"background:transparent; color:{C.TEXT_FAINT}; font-size:14px;")
        self.video_lbl.setText(tr("waiting_conn"))

        self.diff_lbl = QLabel(self.img_container)
        self.diff_lbl.setAlignment(Qt.AlignCenter)
        self.diff_lbl.setStyleSheet(
            f"background:#0a1020; border:1px solid {C.BORDER_HL};"
            f" border-radius:5px; color:{C.TEXT_FAINT}; font-size:11px;")
        self.diff_lbl.setText(tr("diff"))
        self.diff_lbl.setVisible(False)

        # The chart is built on first use.  Twenty eagerly-constructed
        # matplotlib canvases add seconds to startup for a view most tiles
        # never show.
        self.chart: Optional[ChartWidget] = None
        self._chart_logger = None

        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.img_container)   # 0 = live
        root.addWidget(self.view_stack, 1)

    # ── device assignment ─────────────────────────────────────

    def device_id(self) -> str:
        return self._device_id

    def set_descriptors(self, descriptors: List[CameraDescriptor],
                        preferred_index: Optional[int] = None) -> None:
        """Rebuild the camera dropdown, keeping the current selection if possible."""
        self._descriptors = list(descriptors)
        previous = self._device_id

        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        self.cam_combo.addItem(tr("unassigned"), "")
        for d in self._descriptors:
            self.cam_combo.addItem(d.display_name(), d.device_id)

        target_row = 0
        if previous:
            for i in range(self.cam_combo.count()):
                if self.cam_combo.itemData(i) == previous:
                    target_row = i
                    break
        elif preferred_index is not None and preferred_index < len(self._descriptors):
            target_row = preferred_index + 1
        self.cam_combo.setCurrentIndex(target_row)
        self.cam_combo.blockSignals(False)

        new_id = self.cam_combo.currentData() or ""
        if new_id != self._device_id:
            self._device_id = new_id
            self.slot_device_changed.emit(self.slot_id, self._device_id)

    def _on_combo_changed(self, _index: int) -> None:
        self._device_id = self.cam_combo.currentData() or ""
        self.set_disconnected()
        self.slot_device_changed.emit(self.slot_id, self._device_id)
        self.refresh_chart_if_visible()

    # ── coverage / alert ──────────────────────────────────────

    def set_alert_threshold(self, value: float) -> None:
        self.alert_spin.blockSignals(True)
        self.alert_spin.setValue(value)
        self.alert_spin.blockSignals(False)
        self._alert_threshold = value

    def _on_alert_threshold_changed(self, value: float) -> None:
        self._alert_threshold = value

    def set_subtraction_checked(self, checked: bool) -> None:
        """Reflect a global subtract-all toggle without re-emitting."""
        self.sub_btn.blockSignals(True)
        self.sub_btn.setChecked(checked)
        self.sub_btn.blockSignals(False)
        self._sub_enabled = checked
        self.sub_btn.setText(tr("sub_on") if checked else tr("subtract"))
        self.diff_lbl.setVisible(checked and self._show_pip)

    def _on_sub_toggled(self, checked: bool) -> None:
        self._sub_enabled = checked
        self.sub_btn.setText(tr("sub_on") if checked else tr("subtract"))
        self.diff_lbl.setVisible(checked and self._show_pip)
        if not checked:
            self.cov_lbl.setText("--")
            self._clear_alert()
        self.subtraction_toggled.emit(self._device_id, checked)

    def set_chart_logger(self, logger) -> None:
        self._chart_logger = logger
        if self.chart is not None:
            self.chart.set_logger(logger)

    def refresh_chart_if_visible(self) -> None:
        if self._chart_mode and self.chart is not None:
            self.chart.refresh()

    def _ensure_chart(self) -> ChartWidget:
        if self.chart is None:
            self.chart = ChartWidget()
            self.chart.set_logger(self._chart_logger)
            self.view_stack.addWidget(self.chart)       # index 1
        return self.chart

    def _toggle_view(self) -> None:
        self._chart_mode = not self._chart_mode
        if self._chart_mode:
            chart = self._ensure_chart()
            self.view_stack.setCurrentWidget(chart)
            chart.refresh()
        else:
            self.view_stack.setCurrentWidget(self.img_container)
        self.view_btn.setText(tr("show_live") if self._chart_mode else tr("show_chart"))

    # ── frame updates ─────────────────────────────────────────

    def update_frame(self, bgr: np.ndarray, coverage_result, acq_fps: float = 0.0) -> None:
        now = time.time()
        if self._last_frame_time > 0:
            fps = 1.0 / max(now - self._last_frame_time, 0.001)
            self.fps_lbl.setText(f"{fps:.1f}f")
        self._last_frame_time = now
        self._last_bgr = bgr

        w, h = self.video_lbl.width(), self.video_lbl.height()
        if w > 0 and h > 0:
            self.video_lbl.setPixmap(to_pixmap(bgr, w, h))

        if self._fullscreen is not None and self._fullscreen.isVisible():
            self._fullscreen.update_frame(bgr)

        fh, fw = bgr.shape[:2]
        self.res_lbl.setText(f"{fw}×{fh}")

        if coverage_result is not None:
            coverage = coverage_result.coverage_percent
            self.cov_lbl.setText(f"{coverage:.1f}%")
            if self._show_pip and self._sub_enabled:
                pw, ph = self.diff_lbl.width(), self.diff_lbl.height()
                if pw > 0 and ph > 0:
                    self.diff_lbl.setPixmap(
                        to_pixmap(coverage_result.diff_image, pw, ph))
            self._update_alert(coverage)
        elif not self._sub_enabled:
            self.cov_lbl.setText("--")

        self._conn_state = "streaming"
        self.status_dot.setStyleSheet(
            f"color:{C.SUCCESS}; font-size:16px; background:transparent;")
        self.status_dot.setToolTip(tr("status_streaming"))

    def set_status(self, text: str, colour: str) -> None:
        """Show a transient state (connecting / reconnecting / error) on the dot."""
        self.status_dot.setStyleSheet(
            f"color:{colour}; font-size:16px; background:transparent;")
        self.status_dot.setToolTip(text)

    def set_disconnected(self) -> None:
        self._conn_state = "stopped"
        self._clear_alert()
        self._last_bgr = None
        self._last_frame_time = 0.0
        self.video_lbl.setPixmap(QPixmap())
        self.video_lbl.setText(tr("waiting_conn"))
        self.diff_lbl.setVisible(False)
        self.res_lbl.setText("----×----")
        self.fps_lbl.setText("--")
        self.cov_lbl.setText("--")
        self.alert_lbl.setText("")
        self.status_dot.setStyleSheet(
            f"color:{C.DANGER}; font-size:16px; background:transparent;")
        self.status_dot.setToolTip(tr("status_stopped"))

    # ── alert flashing ────────────────────────────────────────

    def _update_alert(self, coverage: float) -> None:
        if coverage >= self._alert_threshold:
            if not self._alert_active:
                self._alert_active = True
                self._flash_timer.start()
            self.alert_lbl.setText(f"⚠ >{self._alert_threshold:.0f}%")
        elif self._alert_active:
            self._clear_alert()

    def _clear_alert(self) -> None:
        self._alert_active = False
        self._flash_timer.stop()
        self._flash_state = False
        self.setStyleSheet(_STYLE_NORMAL)
        self.alert_lbl.setText("")

    def _on_flash_tick(self) -> None:
        self._flash_state = not self._flash_state
        self.setStyleSheet(_STYLE_FLASH_ON if self._flash_state else _STYLE_FLASH_OFF)

    # ── geometry / interaction ────────────────────────────────

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        cw, ch = self.img_container.width(), self.img_container.height()
        self.video_lbl.setGeometry(0, 0, cw, ch)
        pip_w = max(int(cw * 0.28), 100)
        pip_h = max(int(pip_w * 3 / 4), 75)
        margin = 8
        self.diff_lbl.setGeometry(cw - pip_w - margin, margin, pip_w, pip_h)

    def mouseDoubleClickEvent(self, event) -> None:
        if self._fullscreen is None or not self._fullscreen.isVisible():
            self._fullscreen = FullscreenDialog(
                tr("fs_title", title=self.cam_combo.currentText()), self)
            if self._last_bgr is not None:
                self._fullscreen.update_frame(self._last_bgr)
            self._fullscreen.show()
        super().mouseDoubleClickEvent(event)

    # ── i18n ──────────────────────────────────────────────────

    def retranslate(self) -> None:
        if self.cam_combo.count() and self.cam_combo.itemData(0) == "":
            self.cam_combo.setItemText(0, tr("unassigned"))
        self.reset_bg_btn.setText(tr("reset_bg"))
        self.reset_bg_btn.setToolTip(tr("reset_bg_tip"))
        self.alert_spin.setToolTip(tr("alert_spin_tip"))
        self.diff_lbl.setText(tr("diff"))
        self.sub_btn.setText(tr("sub_on") if self._sub_enabled else tr("subtract"))
        self.view_btn.setText(tr("show_live") if self._chart_mode else tr("show_chart"))

        if self._conn_state == "streaming":
            self.status_dot.setToolTip(tr("status_streaming"))
        elif self._conn_state == "stopped":
            self.video_lbl.setText(tr("waiting_conn"))
            self.status_dot.setToolTip(tr("status_stopped"))
        else:
            self.video_lbl.setText(tr("waiting_conn"))
            self.status_dot.setToolTip(tr("status_disconnected"))

        if self.chart is not None:
            self.chart.retranslate()


def to_pixmap(bgr: np.ndarray, w: int, h: int) -> QPixmap:
    """BGR ndarray → aspect-preserving QPixmap scaled into (w, h)."""
    fh, fw = bgr.shape[:2]
    scale = min(w / fw, h / fh)
    nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
    scaled = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, nw, nh, nw * 3, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)
