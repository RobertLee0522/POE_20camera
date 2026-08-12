# -*- coding: utf-8 -*-
"""
ui/splash.py
============
A lightweight, dependency-free startup splash screen.

Drawn entirely with QPainter (no image asset required) so it always matches
``ui/theme.py``'s palette and never breaks if ``assets/icon.png`` moves. Used
by ``app.py`` to give the user visible progress while Aravis discovery, Qt
plugin setup and the main window are being built — the part of startup that
has no window to show yet.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QLinearGradient, QPainter, QPixmap
from PyQt5.QtWidgets import QSplashScreen

from .theme import C

_W, _H = 560, 340


def _build_pixmap() -> QPixmap:
    pix = QPixmap(_W, _H)
    pix.fill(Qt.transparent)

    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)

    # background card with a subtle vertical gradient (BG -> BG_ELEV)
    grad = QLinearGradient(0, 0, 0, _H)
    grad.setColorAt(0.0, QColor(C.BG))
    grad.setColorAt(1.0, QColor(C.BG_ELEV))
    p.setPen(Qt.NoPen)
    p.setBrush(grad)
    p.drawRoundedRect(0, 0, _W, _H, 24, 24)
    p.setPen(QColor(C.BORDER))
    p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(1, 1, _W - 2, _H - 2, 24, 24)

    # 4x5 camera-tile grid motif, same idea as the app icon
    cols, rows = 4, 5
    pad_x, top = 56, 46
    grid_w = _W - 2 * pad_x
    cell_w = grid_w / cols - 10
    cell_h = 26
    gap = 10
    highlight = {(1, 1): C.ACCENT_2, (3, 2): C.SUCCESS, (0, 4): C.ACCENT}
    for r in range(rows):
        for c in range(cols):
            x0 = pad_x + c * (cell_w + gap)
            y0 = top + r * (cell_h + gap)
            color = QColor(highlight.get((c, r), C.SURFACE))
            p.setBrush(color)
            p.setPen(QColor(C.BORDER))
            p.drawRoundedRect(int(x0), int(y0), int(cell_w), int(cell_h), 6, 6)

    # title
    p.setPen(QColor(C.TEXT))
    title_font = QFont("Noto Sans CJK TC", 20, QFont.Bold)
    p.setFont(title_font)
    p.drawText(0, 250, _W, 32, Qt.AlignHCenter, "POE 20-Camera Monitor")

    p.setPen(QColor(C.TEXT_DIM))
    sub_font = QFont("Noto Sans CJK TC", 10)
    p.setFont(sub_font)
    p.drawText(0, 280, _W, 20, Qt.AlignHCenter, "Aravis Edition · GenICam / GigE Vision")

    p.end()
    return pix


class AppSplash(QSplashScreen):
    """Splash screen with a themed message area at the bottom of the card."""

    def __init__(self) -> None:
        super().__init__(_build_pixmap())
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self._msg_color = QColor(C.ACCENT_2)

    def step(self, message: str) -> None:
        """Update the status line and force an immediate repaint."""
        self.showMessage(message, Qt.AlignHCenter | Qt.AlignBottom, self._msg_color)
        # showMessage() alone only queues a repaint; startup keeps the event
        # loop busy right after this, so force it to appear now.
        self.repaint()
        from PyQt5.QtWidgets import QApplication
        QApplication.instance().processEvents()
