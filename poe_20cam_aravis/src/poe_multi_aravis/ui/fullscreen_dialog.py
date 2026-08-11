# -*- coding: utf-8 -*-
"""
ui/fullscreen_dialog.py
========================
Fullscreen preview dialog opened by double-clicking the video widget.
Closed by Esc or double-click.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QSizePolicy

from .i18n import tr


class FullscreenDialog(QDialog):
    def __init__(self, title: str = "", parent=None) -> None:
        super().__init__(parent, Qt.Window)
        self.setWindowTitle(title or tr("waiting_img"))
        self.setStyleSheet("background:#000;")
        self.showMaximized()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._img_lbl = QLabel()
        self._img_lbl.setAlignment(Qt.AlignCenter)
        self._img_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._img_lbl.setText(tr("waiting_img"))
        self._img_lbl.setStyleSheet("color:#333; font-size:20px;")
        layout.addWidget(self._img_lbl)

        hint = QLabel(tr("fs_hint"))
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color:#222; font-size:12px; padding:4px;")
        layout.addWidget(hint)

    def update_frame(self, bgr: np.ndarray) -> None:
        if bgr is None:
            return
        w  = self._img_lbl.width()
        h  = self._img_lbl.height()
        if w <= 0 or h <= 0:
            return
        fh, fw = bgr.shape[:2]
        scale  = min(w / fw, h / fh)
        nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
        scaled = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb    = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
        qimg   = QImage(rgb.data, nw, nh, nw * 3, QImage.Format_RGB888).copy()
        self._img_lbl.setPixmap(QPixmap.fromImage(qimg))

    def mouseDoubleClickEvent(self, event) -> None:
        self.close()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()
        super().keyPressEvent(event)
