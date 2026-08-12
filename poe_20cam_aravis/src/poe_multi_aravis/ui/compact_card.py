# -*- coding: utf-8 -*-
"""
ui/compact_card.py
==================
Read-only camera card for the 4×5 overview grid.

    ┌ #01  Hikrobot MV-CA…  [192.168.1.11]   23.4%  12f  ● ┐
    │  live video (double-click → fullscreen)              │
    └──────────────────────────────────────────────────────┘

The border flashes red at 400 ms whenever coverage crosses the alert
threshold, so an operator watching all 20 feeds at once still sees which
one needs attention.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from .camera_tile import to_pixmap
from .fullscreen_dialog import FullscreenDialog
from .i18n import tr
from .scaling import sf, sx
from .theme import C


class CompactCameraCard(QFrame):
    """Small live tile used in the overview matrix."""

    _STYLE_NORMAL = f"""
        CompactCameraCard {{
            border: 1px solid {C.BORDER};
            border-radius: 8px;
            background: {C.SURFACE};
        }}
    """
    _STYLE_FLASH_ON = """
        CompactCameraCard {
            border: 3px solid #ff2b2b;
            border-radius: 8px;
            background: #1e0606;
        }
    """
    _STYLE_FLASH_OFF = """
        CompactCameraCard {
            border: 3px solid #5c1010;
            border-radius: 8px;
            background: #170404;
        }
    """

    def __init__(self, index: int, label: str = "", device_id: str = "",
                 alert_threshold: float = 50.0, parent=None) -> None:
        super().__init__(parent)
        self.index = index
        self.device_id = device_id
        self._alert_threshold = alert_threshold
        self._alert_active = False
        self._flash_state = False
        self._last_bgr: Optional[np.ndarray] = None
        self._last_frame_time = 0.0
        self._fullscreen: Optional[FullscreenDialog] = None

        self._flash_timer = QTimer(self)
        self._flash_timer.setInterval(400)
        self._flash_timer.timeout.connect(self._on_flash_tick)

        self._build_ui(label)
        self.setStyleSheet(self._STYLE_NORMAL)

    # ── construction ──────────────────────────────────────────

    def _build_ui(self, label: str) -> None:
        self.setFrameShape(QFrame.Box)
        root = QVBoxLayout(self)
        root.setContentsMargins(sx(5), sx(4), sx(5), sx(5))
        root.setSpacing(sx(3))

        bar = QHBoxLayout()
        bar.setSpacing(sx(4))
        bar.setContentsMargins(0, 0, 0, 0)

        id_lbl = QLabel(f"#{self.index + 1:02d}")
        id_lbl.setFixedWidth(sx(26))
        id_lbl.setStyleSheet(
            f"color:{C.ACCENT}; font-weight:700; font-size:{sf(11)}px; background:transparent;")
        bar.addWidget(id_lbl)

        self.name_lbl = QLabel(label)
        self.name_lbl.setStyleSheet(
            f"color:{C.TEXT_DIM}; font-size:{sf(10)}px; background:transparent;")
        bar.addWidget(self.name_lbl, 1)

        self.cov_lbl = QLabel("--")
        self.cov_lbl.setFixedWidth(sx(44))
        self.cov_lbl.setStyleSheet(
            f"color:{C.ACCENT_2}; font-size:{sf(10)}px; font-weight:700;"
            " background:transparent; qproperty-alignment:AlignRight;")
        bar.addWidget(self.cov_lbl)

        self.fps_lbl = QLabel("--")
        self.fps_lbl.setFixedWidth(sx(30))
        self.fps_lbl.setStyleSheet(
            f"color:{C.TEXT_FAINT}; font-size:{sf(10)}px;"
            " background:transparent; qproperty-alignment:AlignRight;")
        bar.addWidget(self.fps_lbl)

        self.dot = QLabel("●")
        self.dot.setFixedWidth(sx(14))
        self.dot.setStyleSheet(
            f"color:{C.DANGER}; font-size:{sf(11)}px; background:transparent;")
        bar.addWidget(self.dot)

        root.addLayout(bar)

        self.img_container = QWidget()
        self.img_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img_container.setStyleSheet("background:#05070c; border-radius:6px;")

        self.video_lbl = QLabel(self.img_container)
        self.video_lbl.setAlignment(Qt.AlignCenter)
        self.video_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_lbl.setStyleSheet(
            f"background:transparent; color:{C.TEXT_FAINT};"
            f" font-size:{sf(15)}px; font-weight:700; letter-spacing:2px;")
        self.video_lbl.setText("OFF")

        root.addWidget(self.img_container, 1)

    # ── public API ────────────────────────────────────────────

    def bind(self, device_id: str, label: str) -> None:
        """Point this card at a (possibly different) camera."""
        if device_id != self.device_id:
            self.set_disconnected()
        self.device_id = device_id
        self.name_lbl.setText(label)

    def set_alert_threshold(self, value: float) -> None:
        self._alert_threshold = value

    def update_frame(self, bgr: np.ndarray, coverage_result) -> None:
        now = time.time()
        if self._last_frame_time > 0:
            fps = 1.0 / max(now - self._last_frame_time, 0.001)
            self.fps_lbl.setText(f"{fps:.0f}f")
        self._last_frame_time = now
        self._last_bgr = bgr

        w, h = self.video_lbl.width(), self.video_lbl.height()
        if w > 0 and h > 0:
            self.video_lbl.setPixmap(to_pixmap(bgr, w, h))

        if self._fullscreen is not None and self._fullscreen.isVisible():
            self._fullscreen.update_frame(bgr)

        if coverage_result is not None:
            coverage = coverage_result.coverage_percent
            self.cov_lbl.setText(f"{coverage:.1f}%")
            if coverage >= self._alert_threshold:
                if not self._alert_active:
                    self._alert_active = True
                    self._flash_timer.start()
            elif self._alert_active:
                self._clear_alert()
        else:
            self.cov_lbl.setText("--")

        self.dot.setStyleSheet(
            f"color:{C.SUCCESS}; font-size:{sf(11)}px; background:transparent;")

    def set_disconnected(self) -> None:
        self._clear_alert()
        self._last_bgr = None
        self._last_frame_time = 0.0
        self.video_lbl.setPixmap(QPixmap())
        self.video_lbl.setText("OFF")
        self.fps_lbl.setText("--")
        self.cov_lbl.setText("--")
        self.dot.setStyleSheet(
            f"color:{C.DANGER}; font-size:{sf(11)}px; background:transparent;")

    # ── internals ─────────────────────────────────────────────

    def _clear_alert(self) -> None:
        self._alert_active = False
        self._flash_timer.stop()
        self._flash_state = False
        self.setStyleSheet(self._STYLE_NORMAL)

    def _on_flash_tick(self) -> None:
        self._flash_state = not self._flash_state
        self.setStyleSheet(
            self._STYLE_FLASH_ON if self._flash_state else self._STYLE_FLASH_OFF)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.video_lbl.setGeometry(
            0, 0, self.img_container.width(), self.img_container.height())

    def mouseDoubleClickEvent(self, event) -> None:
        if self._fullscreen is None or not self._fullscreen.isVisible():
            self._fullscreen = FullscreenDialog(
                tr("fs_title", title=self.name_lbl.text()), self)
            if self._last_bgr is not None:
                self._fullscreen.update_frame(self._last_bgr)
            self._fullscreen.show()
        super().mouseDoubleClickEvent(event)
