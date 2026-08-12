# -*- coding: utf-8 -*-
"""
ui/scaling.py
==============
Screen-aware UI scaling.

The layout (sidebar width, per-tile toolbar, button paddings, font sizes)
was tuned against a spacious reference resolution. On a normal desktop
monitor — even a plain 1920x1080 — two camera tiles side by side plus the
sidebar need more horizontal room than the toolbar row was given, so
controls at the right edge (the status dot, Subtract button, …) get pushed
past the visible screen with no way to reach them.

``init_scale()`` measures the *available* screen geometry (desktop minus
taskbars/docks) once at startup, on whichever screen the app is about to
open on, and derives a single factor that every fixed pixel size in the UI
is multiplied through — so the whole interface shrinks or grows in step
when it is opened on a smaller or bigger display.

Call ``init_scale(app)`` exactly once, right after the ``QApplication`` is
constructed and before any widget is built, so every module that imports
``sx``/``sf`` at class-construction time sees the right factor.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QApplication

# Reference resolution the fixed pixel budget (toolbar widths, card
# paddings, ...) was sized against — the resolution at which scale() == 1.0
# and nothing needs to shrink.
_REF_W, _REF_H = 2400.0, 1350.0

# Clamp so text never becomes unreadable on a tiny screen, and controls
# never balloon absurdly large on an ultra-wide/4K one.
_MIN_SCALE, _MAX_SCALE = 0.60, 1.30
# Fonts get a gentler floor than chrome (buttons/labels) so text stays
# legible even when the layout itself has to shrink hard.
_MIN_FONT_SCALE = 0.78

_scale = 1.0
_screen_w, _screen_h = 1920, 1080


def init_scale(app: QApplication) -> float:
    """Measure the primary/active screen and set the global scale factor."""
    global _scale, _screen_w, _screen_h
    screen = app.primaryScreen()
    if screen is None:
        return _scale
    geo = screen.availableGeometry()
    if geo.width() > 0 and geo.height() > 0:
        _screen_w, _screen_h = geo.width(), geo.height()
        factor = min(_screen_w / _REF_W, _screen_h / _REF_H)
        _scale = max(_MIN_SCALE, min(_MAX_SCALE, factor))
    return _scale


def scale() -> float:
    return _scale


def screen_size() -> tuple:
    """(width, height) of the available area on the screen used at startup."""
    return _screen_w, _screen_h


def sx(px: float) -> int:
    """Scale a chrome dimension: widths, heights, spacing, radii, padding."""
    return max(1, round(px * _scale))


def sf(px: float) -> int:
    """Scale a font size, with a gentler floor so text stays legible."""
    return max(1, round(px * max(_scale, _MIN_FONT_SCALE)))
