# -*- coding: utf-8 -*-
"""
ui/theme.py
===========
Modern flat dark design system for the single-camera app.

Provides a colour palette, a global Qt stylesheet, and small helpers for
building the card-based layout used across the UI.  Keeping all styling in one
place makes the "modernized redesign" consistent and easy to tweak.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout, QGraphicsDropShadowEffect
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt


class C:
    """Colour palette (modern flat dark)."""
    BG          = "#0b0e14"   # window background
    BG_ELEV     = "#111621"   # elevated background (sidebar)
    SURFACE     = "#161c27"   # card surface
    SURFACE_2   = "#1c2332"   # nested surface / inputs
    BORDER      = "#242c3a"   # hairline borders
    BORDER_HL   = "#3b4a63"   # focused/hover border

    TEXT        = "#e6edf3"   # primary text
    TEXT_DIM    = "#8b98a9"   # secondary text
    TEXT_FAINT  = "#5b6675"   # tertiary / placeholder

    ACCENT      = "#3b82f6"   # primary accent (blue)
    ACCENT_2    = "#22d3ee"   # cyan highlight
    SUCCESS     = "#34d399"   # green
    WARNING     = "#fbbf24"   # amber
    DANGER      = "#f87171"   # red
    VIOLET      = "#a78bfa"


def app_stylesheet() -> str:
    """Global application stylesheet."""
    return f"""
    QWidget {{
        background: transparent;
        color: {C.TEXT};
        font-family: "Noto Sans CJK TC","Microsoft JhengHei","Segoe UI",sans-serif;
        font-size: 13px;
    }}
    QMainWindow, #RootBg {{ background: {C.BG}; }}
    #Sidebar {{ background: {C.BG_ELEV}; border-right: 1px solid {C.BORDER}; }}

    QScrollArea {{ border: none; background: transparent; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {C.BORDER_HL}; border-radius: 5px; min-height: 30px; }}
    QScrollBar::handle:vertical:hover {{ background: {C.ACCENT}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    /* Inputs */
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
        background: {C.SURFACE_2};
        border: 1px solid {C.BORDER};
        border-radius: 8px;
        padding: 5px 9px;
        color: {C.TEXT};
        selection-background-color: {C.ACCENT};
        min-height: 18px;
    }}
    QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {C.BORDER_HL}; }}
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: {C.ACCENT}; }}
    QComboBox::drop-down {{ border: none; width: 20px; }}
    QComboBox::down-arrow {{ image: none; border-left: 4px solid transparent;
        border-right: 4px solid transparent; border-top: 5px solid {C.TEXT_DIM}; margin-right: 8px; }}
    QComboBox QAbstractItemView {{
        background: {C.SURFACE}; border: 1px solid {C.BORDER};
        selection-background-color: {C.ACCENT}; outline: none; border-radius: 8px;
    }}
    QSpinBox::up-button, QSpinBox::down-button,
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        width: 16px; border: none; background: transparent;
    }}

    /* Sliders */
    QSlider::groove:horizontal {{ height: 5px; background: {C.SURFACE_2}; border-radius: 3px; }}
    QSlider::sub-page:horizontal {{ background: {C.ACCENT}; border-radius: 3px; }}
    QSlider::handle:horizontal {{
        background: {C.TEXT}; width: 15px; height: 15px; margin: -6px 0;
        border-radius: 8px; border: 2px solid {C.ACCENT};
    }}

    /* Checkboxes / radios */
    QCheckBox, QRadioButton {{ color: {C.TEXT_DIM}; spacing: 7px; }}
    QCheckBox::indicator, QRadioButton::indicator {{ width: 16px; height: 16px; }}
    QCheckBox::indicator {{ border: 1px solid {C.BORDER_HL}; border-radius: 5px; background: {C.SURFACE_2}; }}
    QCheckBox::indicator:checked {{ background: {C.ACCENT}; border-color: {C.ACCENT}; }}
    QRadioButton::indicator {{ border: 1px solid {C.BORDER_HL}; border-radius: 8px; background: {C.SURFACE_2}; }}
    QRadioButton::indicator:checked {{ background: {C.ACCENT}; border: 4px solid {C.SURFACE_2}; }}

    QToolTip {{ background: {C.SURFACE}; color: {C.TEXT}; border: 1px solid {C.BORDER_HL};
        border-radius: 6px; padding: 4px 8px; }}
    """


# ── button style factories ────────────────────────────────────

def btn_primary() -> str:
    return f"""
    QPushButton {{ background: {C.ACCENT}; color: white; border: none;
        border-radius: 8px; padding: 8px 14px; font-weight: 600; }}
    QPushButton:hover:enabled {{ background: #4f8ff7; }}
    QPushButton:pressed {{ background: #2f6fd6; }}
    QPushButton:disabled {{ background: {C.SURFACE_2}; color: {C.TEXT_FAINT}; }}
    """


def btn_ghost() -> str:
    return f"""
    QPushButton {{ background: {C.SURFACE_2}; color: {C.TEXT}; border: 1px solid {C.BORDER};
        border-radius: 8px; padding: 8px 12px; }}
    QPushButton:hover:enabled {{ background: {C.SURFACE}; border-color: {C.BORDER_HL}; }}
    QPushButton:pressed {{ background: #10151f; }}
    QPushButton:disabled {{ color: {C.TEXT_FAINT}; }}
    """


def btn_success() -> str:
    return f"""
    QPushButton {{ background: {C.SUCCESS}; color: #04231a; border: none;
        border-radius: 8px; padding: 8px 14px; font-weight: 600; }}
    QPushButton:hover:enabled {{ background: #4ce3a9; }}
    QPushButton:disabled {{ background: {C.SURFACE_2}; color: {C.TEXT_FAINT}; }}
    """


def btn_danger() -> str:
    return f"""
    QPushButton {{ background: {C.DANGER}; color: #2a0808; border: none;
        border-radius: 8px; padding: 8px 14px; font-weight: 600; }}
    QPushButton:hover:enabled {{ background: #fb8b8b; }}
    QPushButton:disabled {{ background: {C.SURFACE_2}; color: {C.TEXT_FAINT}; }}
    """


def btn_toggle() -> str:
    """Checkable pill button (e.g. subtraction on/off)."""
    return f"""
    QPushButton {{ background: {C.SURFACE_2}; color: {C.TEXT_DIM}; border: 1px solid {C.BORDER};
        border-radius: 8px; padding: 8px 12px; font-weight: 600; }}
    QPushButton:hover {{ border-color: {C.BORDER_HL}; }}
    QPushButton:checked {{ background: rgba(52,211,153,0.16); color: {C.SUCCESS};
        border: 1px solid {C.SUCCESS}; }}
    """


# ── card container ────────────────────────────────────────────

class Card(QFrame):
    """A titled surface card with a subtle border and soft shadow."""

    def __init__(self, title: str = "", icon: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.setStyleSheet(
            f"#Card {{ background: {C.SURFACE}; border: 1px solid {C.BORDER};"
            f" border-radius: 12px; }}")
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(18)
        shadow.setXOffset(0)
        shadow.setYOffset(3)
        shadow.setColor(QColor(0, 0, 0, 90))
        self.setGraphicsEffect(shadow)

        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(14, 12, 14, 14)
        self._lay.setSpacing(10)

        self._title_label = None
        if title:
            self._title_label = QLabel(f"{icon}  {title}".strip())
            self._title_label.setStyleSheet(
                f"color: {C.TEXT_DIM}; font-size: 12px; font-weight: 700;"
                f" letter-spacing: 0.5px; background: transparent; border: none;")
            self._lay.addWidget(self._title_label)
        self._title_text = title
        self._icon = icon

    def set_title(self, title: str) -> None:
        self._title_text = title
        if self._title_label:
            self._title_label.setText(f"{self._icon}  {title}".strip())

    def body(self) -> QVBoxLayout:
        return self._lay

    def add(self, widget) -> None:
        self._lay.addWidget(widget)

    def add_layout(self, layout) -> None:
        self._lay.addLayout(layout)


def hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"background: {C.BORDER}; max-height: 1px; border: none;")
    return line


def key_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {C.TEXT_DIM}; font-size: 12px; background: transparent;")
    return lbl


def value_label(text: str = "–") -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {C.TEXT}; font-size: 13px; font-weight: 600; background: transparent;")
    return lbl
