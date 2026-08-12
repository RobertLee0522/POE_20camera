# -*- coding: utf-8 -*-
"""
ui/main_window.py
=================
Main window for the 20-camera Aravis monitor.

    ┌ sidebar (340px, scrollable) ┬ tabbed camera pages (2 per page)   ┐
    │  hardware monitor           │  ── or ──                          │
    │  camera control             │  4×5 overview matrix               │
    │  global stream params       │                                    │
    │  camera parameters          │                                    │
    │  detected devices           │                                    │
    └─────────────────────────────┴────────────────────────────────────┘

The window is a pure view: every action goes through ``MultiCameraService``
and every update arrives as a Qt signal on the main thread.  No Aravis,
torch or OpenCV pipeline code lives here.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QButtonGroup, QDoubleSpinBox, QFormLayout, QGridLayout, QHBoxLayout,
    QLabel, QListWidget, QMainWindow, QMessageBox, QPushButton, QRadioButton,
    QScrollArea, QSlider, QSpinBox, QSplitter, QStatusBar, QTabWidget,
    QVBoxLayout, QWidget,
)

from ..domain.camera_state import CameraState
from ..domain.models import CameraDescriptor
from ..services.gpu_batch_processor import cuda_available
from ..services.multi_camera_service import MultiCameraService
from ..settings import Settings
from .camera_tile import CameraTile
from .compact_card import CompactCameraCard
from .i18n import get_language, set_language, toggle_language, tr
from .scaling import screen_size, sf, sx
from .theme import (
    C, Card, app_stylesheet, btn_danger, btn_ghost, btn_primary, btn_success,
    btn_toggle, hline, key_label, value_label,
)

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Top-level window: sidebar + tabbed camera pages + overview matrix."""

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self._settings = settings
        set_language(settings.language)

        self.max_cameras = settings.max_cameras
        self.per_page = settings.cameras_per_page
        self.page_count = max(1, (self.max_cameras + self.per_page - 1) // self.per_page)
        self.ov_cols = settings.overview_columns

        self.service = MultiCameraService(settings, parent=self)

        self._tiles: List[CameraTile] = []
        self._cards: List[CompactCameraCard] = []
        self._tile_by_device: Dict[str, CameraTile] = {}
        self._card_by_device: Dict[str, CompactCameraCard] = {}
        self._descriptors: List[CameraDescriptor] = []

        self.setWindowTitle(tr("win_title", n=self.max_cameras))
        # Restore-size when un-maximized: the actual screen's available
        # area, not a hardcoded 1920x1080 that may not fit the display the
        # app happens to open on.
        self.resize(*screen_size())
        self.setStyleSheet(app_stylesheet())

        self._build_ui()
        self._connect_service()

        self.service.start_background_services()
        QTimer.singleShot(0, self.refresh_devices)
        if settings.auto_start_all:
            QTimer.singleShot(800, self.start_all)

    # ══════════════════════════════════════════════════════════
    #  construction
    # ══════════════════════════════════════════════════════════

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("RootBg")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_camera_area())
        splitter.setStretchFactor(1, 1)
        sw, _ = screen_size()
        splitter.setSizes([sx(340), max(sx(1580), sw - sx(340))])
        outer.addWidget(splitter)

        self.setStatusBar(QStatusBar())
        self.statusBar().setStyleSheet(
            f"color:{C.TEXT_DIM}; background:{C.BG_ELEV};"
            f" border-top:1px solid {C.BORDER};")
        self.statusBar().showMessage(tr("param_ready"))

    # ── sidebar ───────────────────────────────────────────────

    def _build_sidebar(self) -> QWidget:
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(sx(12), sx(12), sx(12), sx(12))
        lay.setSpacing(sx(12))

        # language toggle
        top = QHBoxLayout()
        title = QLabel("POE · Aravis")
        title.setStyleSheet(
            f"color:{C.TEXT}; font-size:{sf(15)}px; font-weight:700; background:transparent;")
        top.addWidget(title)
        top.addStretch()
        self.lang_btn = QPushButton(tr("lang_btn"))
        self.lang_btn.setFixedSize(sx(66), sx(30))
        self.lang_btn.setToolTip(tr("lang_tip"))
        self.lang_btn.setStyleSheet(btn_ghost())
        self.lang_btn.clicked.connect(self._toggle_language)
        top.addWidget(self.lang_btn)
        lay.addLayout(top)

        lay.addWidget(self._build_hw_card())
        lay.addWidget(self._build_control_card())
        lay.addWidget(self._build_params_card())
        lay.addWidget(self._build_cam_param_card())
        lay.addWidget(self._build_devices_card())
        lay.addStretch()

        scroll = QScrollArea()
        scroll.setObjectName("Sidebar")
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(sx(348))
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll

    def _build_hw_card(self) -> Card:
        self.hw_card = Card(tr("hw_monitor"), "🖥")
        grid = QGridLayout()
        grid.setHorizontalSpacing(sx(10))
        grid.setVerticalSpacing(sx(6))

        self.cpu_val  = value_label("–")
        self.ram_val  = value_label("–")
        self.gpu_val  = value_label("–")
        self.vram_val = value_label("–")
        self.cam_cnt_val = value_label(tr("streaming_n", n=0))

        self._hw_keys = [key_label(tr("cpu")), key_label(tr("ram")),
                         key_label(tr("gpu")), key_label(tr("vram"))]
        for row, (k, v) in enumerate(zip(
                self._hw_keys, [self.cpu_val, self.ram_val, self.gpu_val, self.vram_val])):
            grid.addWidget(k, row, 0)
            grid.addWidget(v, row, 1)
        self.hw_card.add_layout(grid)
        self.hw_card.add(hline())
        self.hw_card.add(self.cam_cnt_val)
        return self.hw_card

    def _build_control_card(self) -> Card:
        self.ctrl_card = Card(tr("cam_control"), "📷")

        self.refresh_btn = QPushButton(tr("refresh_dev"))
        self.refresh_btn.setMinimumHeight(sx(34))
        self.refresh_btn.setStyleSheet(btn_ghost())
        self.refresh_btn.clicked.connect(self.refresh_devices)
        self.ctrl_card.add(self.refresh_btn)

        self.mode_lbl = key_label(tr("compute_mode"))
        self.ctrl_card.add(self.mode_lbl)
        mode_row = QHBoxLayout()
        self.cpu_radio = QRadioButton("CPU")
        self.cuda_radio = QRadioButton("CUDA (GPU)")
        self._has_cuda = cuda_available()
        if not self._has_cuda:
            self.cuda_radio.setEnabled(False)
            self.cuda_radio.setToolTip(tr("cuda_na"))
        if self.service.processing_mode == "cuda" and self._has_cuda:
            self.cuda_radio.setChecked(True)
        else:
            self.cpu_radio.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.cpu_radio)
        group.addButton(self.cuda_radio)
        self.cpu_radio.toggled.connect(self._on_mode_changed)
        mode_row.addWidget(self.cpu_radio)
        mode_row.addWidget(self.cuda_radio)
        self.ctrl_card.add_layout(mode_row)

        self.start_all_btn = QPushButton(tr("start_all"))
        self.start_all_btn.setMinimumHeight(sx(36))
        self.start_all_btn.setStyleSheet(btn_success())
        self.start_all_btn.clicked.connect(self.start_all)
        self.ctrl_card.add(self.start_all_btn)

        self.stop_all_btn = QPushButton(tr("stop_all"))
        self.stop_all_btn.setMinimumHeight(sx(36))
        self.stop_all_btn.setStyleSheet(btn_danger())
        self.stop_all_btn.clicked.connect(self.stop_all)
        self.ctrl_card.add(self.stop_all_btn)
        return self.ctrl_card

    def _build_params_card(self) -> Card:
        self.param_card = Card(tr("global_params"), "⚙")

        self.fps_lbl = key_label(tr("fps_limit"))
        self.param_card.add(self.fps_lbl)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(self._settings.target_fps)
        self.fps_spin.setMinimumHeight(sx(30))
        self.fps_spin.valueChanged.connect(self.service.set_target_fps)
        self.param_card.add(self.fps_spin)

        self.thresh_lbl = key_label(tr("threshold"))
        self.param_card.add(self.thresh_lbl)
        th_row = QHBoxLayout()
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(1, 100)
        self.thresh_slider.setValue(self._settings.difference_threshold)
        self.thresh_val = value_label(str(self._settings.difference_threshold))
        self.thresh_val.setFixedWidth(sx(34))
        self.thresh_slider.valueChanged.connect(self._on_threshold_changed)
        th_row.addWidget(self.thresh_slider, 1)
        th_row.addWidget(self.thresh_val)
        self.param_card.add_layout(th_row)

        self.sub_all_btn = QPushButton(tr("sub_all_off"))
        self.sub_all_btn.setCheckable(True)
        self.sub_all_btn.setMinimumHeight(sx(34))
        self.sub_all_btn.setStyleSheet(btn_toggle())
        self.sub_all_btn.clicked.connect(self._toggle_all_subtraction)
        self.param_card.add(self.sub_all_btn)

        self.reset_all_bg_btn = QPushButton(tr("reset_all_bg"))
        self.reset_all_bg_btn.setMinimumHeight(sx(34))
        self.reset_all_bg_btn.setToolTip(tr("reset_all_bg_tip"))
        self.reset_all_bg_btn.setStyleSheet(btn_ghost())
        self.reset_all_bg_btn.clicked.connect(self.service.reset_all_backgrounds)
        self.param_card.add(self.reset_all_bg_btn)

        self.param_card.add(hline())

        self.alert_lbl = key_label(tr("alert_hdr"))
        self.param_card.add(self.alert_lbl)
        alert_row = QHBoxLayout()
        self.global_alert_spin = QDoubleSpinBox()
        self.global_alert_spin.setRange(0.0, 100.0)
        self.global_alert_spin.setValue(self._settings.alert_threshold_percent)
        self.global_alert_spin.setSuffix("%")
        self.global_alert_spin.setDecimals(1)
        self.global_alert_spin.setMinimumHeight(sx(30))
        self.global_alert_spin.setToolTip(tr("global_alert_tip"))
        self.apply_alert_btn = QPushButton(tr("apply_all"))
        self.apply_alert_btn.setFixedWidth(sx(104))
        self.apply_alert_btn.setMinimumHeight(sx(30))
        self.apply_alert_btn.setStyleSheet(btn_ghost())
        self.apply_alert_btn.clicked.connect(
            lambda: self._apply_global_alert(self.global_alert_spin.value()))
        alert_row.addWidget(self.global_alert_spin, 1)
        alert_row.addWidget(self.apply_alert_btn)
        self.param_card.add_layout(alert_row)
        return self.param_card

    def _build_cam_param_card(self) -> Card:
        self.cam_param_card = Card(tr("cam_params"), "🎛")

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setHorizontalSpacing(sx(8))
        form.setVerticalSpacing(sx(7))
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.exp_key  = key_label(tr("exposure"))
        self.gain_key = key_label(tr("gain"))
        self.fr_key   = key_label(tr("frame_rate"))

        self.exp_spin = QDoubleSpinBox()
        self.exp_spin.setRange(1.0, 10_000_000.0)
        self.exp_spin.setDecimals(1)
        self.exp_spin.setSingleStep(1000.0)
        self.exp_spin.setValue(20000.0)
        self.exp_spin.setMinimumHeight(sx(28))

        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.0, 48.0)
        self.gain_spin.setDecimals(2)
        self.gain_spin.setSingleStep(0.5)
        self.gain_spin.setMinimumHeight(sx(28))

        self.fr_spin = QDoubleSpinBox()
        self.fr_spin.setRange(0.1, 200.0)
        self.fr_spin.setDecimals(2)
        self.fr_spin.setSingleStep(1.0)
        self.fr_spin.setValue(15.0)
        self.fr_spin.setMinimumHeight(sx(28))

        form.addRow(self.exp_key,  self.exp_spin)
        form.addRow(self.gain_key, self.gain_spin)
        form.addRow(self.fr_key,   self.fr_spin)
        self.cam_param_card.add_layout(form)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(sx(6))
        self.get_param_btn = QPushButton(tr("get_param"))
        self.get_param_btn.setMinimumHeight(sx(30))
        self.get_param_btn.setStyleSheet(btn_ghost())
        self.get_param_btn.clicked.connect(self._get_cam_params)
        self.set_param_btn = QPushButton(tr("set_param"))
        self.set_param_btn.setMinimumHeight(sx(30))
        self.set_param_btn.setStyleSheet(btn_primary())
        self.set_param_btn.clicked.connect(self._set_cam_params)
        btn_row.addWidget(self.get_param_btn)
        btn_row.addWidget(self.set_param_btn)
        self.cam_param_card.add_layout(btn_row)

        self.cam_param_status = QLabel(tr("param_ready"))
        self.cam_param_status.setAlignment(Qt.AlignCenter)
        self.cam_param_status.setStyleSheet(
            f"color:{C.TEXT_FAINT}; font-size:{sf(11)}px; background:transparent;")
        self.cam_param_card.add(self.cam_param_status)
        return self.cam_param_card

    def _build_devices_card(self) -> Card:
        self.dev_card = Card(tr("detected_dev"), "📋")
        self.dev_list = QListWidget()
        self.dev_list.setMinimumHeight(sx(140))
        self.dev_list.setStyleSheet(
            f"QListWidget {{ background:{C.SURFACE_2}; border:1px solid {C.BORDER};"
            f" border-radius:{sx(8)}px; font-size:{sf(12)}px; }}"
            f" QListWidget::item {{ padding:{sx(4)}px {sx(6)}px; }}"
            f" QListWidget::item:selected {{ background:{C.ACCENT}; }}")
        self.dev_card.add(self.dev_list)
        return self.dev_card

    # ── camera area ───────────────────────────────────────────

    def _build_camera_area(self) -> QWidget:
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(sx(8), sx(8), sx(8), sx(8))
        lay.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {C.BORDER};
                background: {C.BG_ELEV};
                border-radius: {sx(10)}px;
            }}
            QTabBar::tab {{
                background: {C.SURFACE};
                color: {C.TEXT_DIM};
                border: 1px solid {C.BORDER};
                padding: {sx(8)}px {sx(14)}px;
                border-top-left-radius: {sx(8)}px;
                border-top-right-radius: {sx(8)}px;
                font-size: {sf(12)}px;
                min-width: {sx(78)}px;
            }}
            QTabBar::tab:selected {{
                background: {C.SURFACE_2};
                color: {C.TEXT};
                border-bottom: 2px solid {C.ACCENT};
                font-weight: 700;
            }}
            QTabBar::tab:hover {{ background: {C.SURFACE_2}; }}
        """)

        for page in range(self.page_count):
            page_widget = QWidget()
            page_layout = QHBoxLayout(page_widget)
            page_layout.setContentsMargins(sx(8), sx(8), sx(8), sx(8))
            page_layout.setSpacing(sx(8))

            for j in range(self.per_page):
                slot_id = page * self.per_page + j
                if slot_id >= self.max_cameras:
                    break
                tile = CameraTile(
                    slot_id, self._descriptors,
                    alert_threshold=self._settings.alert_threshold_percent,
                    show_pip=self._settings.show_difference_pip,
                )
                tile.subtraction_toggled.connect(self._on_tile_subtraction)
                tile.slot_device_changed.connect(self._on_slot_device_changed)
                tile.reset_bg_requested.connect(self._on_reset_bg)
                page_layout.addWidget(tile, 1)
                self._tiles.append(tile)

            first = page * self.per_page + 1
            last = min((page + 1) * self.per_page, self.max_cameras)
            self.tabs.addTab(page_widget, tr("page_tab", p=page + 1, a=first, b=last))

        self.overview_btn = QPushButton(tr("overview"))
        self.overview_btn.setCheckable(True)
        self.overview_btn.setFixedHeight(sx(28))
        self.overview_btn.setStyleSheet(btn_ghost())
        self.overview_btn.clicked.connect(self._toggle_overview)
        self.tabs.setCornerWidget(self.overview_btn, Qt.TopRightCorner)
        lay.addWidget(self.tabs)

        # ── overview matrix ──
        self.overview = QWidget()
        self.overview.setVisible(False)
        ov_root = QVBoxLayout(self.overview)
        ov_root.setContentsMargins(sx(6), sx(6), sx(6), sx(6))
        ov_root.setSpacing(sx(6))

        header = QHBoxLayout()
        self.ov_title = QLabel(tr("overview_title"))
        self.ov_title.setStyleSheet(
            f"color:{C.TEXT}; font-size:{sf(14)}px; font-weight:700; background:transparent;")
        header.addWidget(self.ov_title)
        header.addStretch()
        self.back_btn = QPushButton(tr("back_to_tabs"))
        self.back_btn.setFixedHeight(sx(28))
        self.back_btn.setStyleSheet(btn_ghost())
        self.back_btn.clicked.connect(lambda: self.overview_btn.click())
        header.addWidget(self.back_btn)
        ov_root.addLayout(header)

        grid = QGridLayout()
        grid.setSpacing(sx(6))
        for idx in range(self.max_cameras):
            card = CompactCameraCard(
                idx, label=f"Cam {idx + 1:02d}",
                alert_threshold=self._settings.alert_threshold_percent)
            self._cards.append(card)
            grid.addWidget(card, idx // self.ov_cols, idx % self.ov_cols)
        ov_root.addLayout(grid, 1)
        lay.addWidget(self.overview)
        return container

    # ══════════════════════════════════════════════════════════
    #  service wiring
    # ══════════════════════════════════════════════════════════

    def _connect_service(self) -> None:
        s = self.service
        s.devices_updated.connect(self._on_devices_updated)
        s.frame_ready.connect(self._on_frame_ready)
        s.state_changed.connect(self._on_state_changed)
        s.hardware_stats.connect(self._on_hardware_stats)
        s.error_message.connect(self._on_error)
        s.info_message.connect(self._on_info)
        s.reconnecting.connect(self._on_reconnecting)
        s.reconnected.connect(self._on_reconnected)
        s.logs_flushed.connect(self._on_logs_flushed)

    # ── devices ───────────────────────────────────────────────

    def refresh_devices(self) -> None:
        self.service.refresh_devices()

    def _on_devices_updated(self, descriptors: List[CameraDescriptor]) -> None:
        self._descriptors = descriptors

        self.dev_list.clear()
        if not descriptors:
            self.dev_list.addItem(tr("no_device"))
        for i, d in enumerate(descriptors):
            self.dev_list.addItem(f"[{i:02d}]  {d.display_name()}")

        for slot, tile in enumerate(self._tiles):
            tile.set_descriptors(descriptors, preferred_index=slot)

        for idx, card in enumerate(self._cards):
            if idx < len(descriptors):
                d = descriptors[idx]
                card.bind(d.device_id, d.display_name())
            else:
                card.bind("", f"Cam {idx + 1:02d}")

        self._rebuild_routing()

    def _rebuild_routing(self) -> None:
        self._tile_by_device = {
            t.device_id(): t for t in self._tiles if t.device_id()
        }
        self._card_by_device = {
            c.device_id: c for c in self._cards if c.device_id
        }

    def _on_slot_device_changed(self, _slot_id: int, device_id: str) -> None:
        self._rebuild_routing()
        tile = self._tile_by_device.get(device_id)
        if tile is not None and device_id:
            tile.set_chart_logger(self.service.logger_for(device_id))

    # ── acquisition ───────────────────────────────────────────

    def start_all(self) -> None:
        if self.cuda_radio.isChecked() and not self._has_cuda:
            QMessageBox.warning(self, tr("cuda_err"), tr("cuda_fallback"))
            self.cpu_radio.setChecked(True)

        # Lock the compute mode while cameras stream: swapping the pipeline
        # under 20 live workers is not worth the failure modes.
        self.cpu_radio.setEnabled(False)
        self.cuda_radio.setEnabled(False)

        self.service.start_all()
        for device_id, tile in self._tile_by_device.items():
            tile.set_chart_logger(self.service.logger_for(device_id))
        self._update_stream_count()

    def stop_all(self) -> None:
        self.service.stop_all()
        self.service.disconnect_all()

        self.cpu_radio.setEnabled(True)
        self.cuda_radio.setEnabled(self._has_cuda)

        for tile in self._tiles:
            tile.set_disconnected()
            tile.set_subtraction_checked(False)
        for card in self._cards:
            card.set_disconnected()
        self.sub_all_btn.setChecked(False)
        self.sub_all_btn.setText(tr("sub_all_off"))
        self._update_stream_count()

    def _on_mode_changed(self, _checked: bool) -> None:
        mode = "cuda" if self.cuda_radio.isChecked() else "cpu"
        effective = self.service.set_processing_mode(mode)
        if effective != mode:
            self.cpu_radio.setChecked(True)

    # ── frames ────────────────────────────────────────────────

    def _on_frame_ready(self, device_id: str, display_bgr: np.ndarray,
                        coverage, _stats) -> None:
        tile = self._tile_by_device.get(device_id)
        if tile is not None:
            tile.update_frame(display_bgr, coverage)
        card = self._card_by_device.get(device_id)
        if card is not None:
            card.update_frame(display_bgr, coverage)

    def _on_state_changed(self, device_id: str, _old, new) -> None:
        tile = self._tile_by_device.get(device_id)
        if tile is None:
            return
        if new == CameraState.STREAMING:
            tile.set_status(tr("status_streaming"), C.SUCCESS)
        elif new == CameraState.CONNECTED:
            tile.set_status(tr("status_connected"), C.WARNING)
        elif new == CameraState.CONNECTING:
            tile.set_status(tr("status_connecting"), C.WARNING)
        elif new == CameraState.ERROR:
            tile.set_status(tr("status_error"), C.DANGER)
        elif new == CameraState.DISCONNECTED:
            tile.set_disconnected()
        self._update_stream_count()

    def _on_reconnecting(self, device_id: str, attempt: int, maximum: int) -> None:
        tile = self._tile_by_device.get(device_id)
        if tile is not None:
            tile.set_status(tr("status_reconnecting", a=attempt, m=maximum), C.WARNING)

    def _on_reconnected(self, device_id: str) -> None:
        tile = self._tile_by_device.get(device_id)
        if tile is not None:
            tile.set_status(tr("status_streaming"), C.SUCCESS)

    # ── subtraction / background ──────────────────────────────

    def _on_tile_subtraction(self, device_id: str, enabled: bool) -> None:
        if not device_id:
            return
        self.service.set_subtraction(device_id, enabled)
        tile = self._tile_by_device.get(device_id)
        if tile is not None:
            tile.set_chart_logger(self.service.logger_for(device_id))

    def _toggle_all_subtraction(self, checked: bool) -> None:
        self.sub_all_btn.setText(tr("sub_all_on") if checked else tr("sub_all_off"))
        self.service.set_subtraction_all(checked)
        for tile in self._tiles:
            tile.set_subtraction_checked(checked)
            if checked and tile.device_id():
                tile.set_chart_logger(self.service.logger_for(tile.device_id()))

    def _on_reset_bg(self, device_id: str) -> None:
        if device_id:
            self.service.reset_background(device_id)

    # ── global params ─────────────────────────────────────────

    def _on_threshold_changed(self, value: int) -> None:
        self.thresh_val.setText(str(value))
        self.service.set_threshold(value)

    def _apply_global_alert(self, value: float) -> None:
        self.service.set_alert_threshold(value)
        for tile in self._tiles:
            tile.set_alert_threshold(value)
        for card in self._cards:
            card.set_alert_threshold(value)

    def _get_cam_params(self) -> None:
        connected = self.service.connected_ids()
        if not connected:
            self._param_status(tr("param_no_cam"), C.WARNING)
            return
        device_id = connected[0]
        params = self.service.read_params(device_id)
        if params.get("exposure"):
            self.exp_spin.setValue(float(params["exposure"]))
        if params.get("gain") is not None:
            self.gain_spin.setValue(float(params["gain"]))
        if params.get("frame_rate"):
            self.fr_spin.setValue(float(params["frame_rate"]))
        desc = self.service.descriptor_for(device_id)
        name = desc.display_name() if desc else device_id
        self._param_status(tr("param_read_ok", name=name), C.SUCCESS)

    def _set_cam_params(self) -> None:
        if not self.service.connected_ids():
            self._param_status(tr("param_no_cam"), C.WARNING)
            return
        ok, failures = self.service.apply_params(
            exposure_us=self.exp_spin.value(),
            gain_db=self.gain_spin.value(),
            frame_rate=self.fr_spin.value(),
        )
        if failures:
            self._param_status(tr("param_failed", items=", ".join(failures)), C.DANGER)
        else:
            self._param_status(tr("param_applied", n=ok), C.SUCCESS)

    def _param_status(self, message: str, colour: str) -> None:
        self.cam_param_status.setText(message)
        self.cam_param_status.setStyleSheet(
            f"color:{colour}; font-size:{sf(11)}px; background:transparent;")

    # ── misc updates ──────────────────────────────────────────

    def _on_hardware_stats(self, cpu: float, ram: float, gpu: str, vram: str) -> None:
        self.cpu_val.setText(f"{cpu:.1f} %")
        self.ram_val.setText(f"{ram:.1f} %")
        self.gpu_val.setText(gpu)
        self.vram_val.setText(vram)
        self._update_stream_count()

    def _update_stream_count(self) -> None:
        self.cam_cnt_val.setText(
            tr("streaming_n", n=self.service.streaming_count()))

    def _on_logs_flushed(self) -> None:
        for tile in self._tiles:
            tile.refresh_chart_if_visible()

    def _on_error(self, message: str) -> None:
        self.statusBar().showMessage(f"⚠  {message}", 8000)
        log.warning("%s", message)

    def _on_info(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)

    def _toggle_overview(self, checked: bool) -> None:
        self.tabs.setVisible(not checked)
        self.overview.setVisible(checked)

    # ── i18n ──────────────────────────────────────────────────

    def _toggle_language(self) -> None:
        toggle_language()
        self._settings.language = get_language()
        self._retranslate()

    def _retranslate(self) -> None:
        self.setWindowTitle(tr("win_title", n=self.max_cameras))
        self.lang_btn.setText(tr("lang_btn"))
        self.lang_btn.setToolTip(tr("lang_tip"))

        self.hw_card.set_title(tr("hw_monitor"))
        for lbl, key in zip(self._hw_keys, ("cpu", "ram", "gpu", "vram")):
            lbl.setText(tr(key))

        self.ctrl_card.set_title(tr("cam_control"))
        self.refresh_btn.setText(tr("refresh_dev"))
        self.mode_lbl.setText(tr("compute_mode"))
        if not self._has_cuda:
            self.cuda_radio.setToolTip(tr("cuda_na"))
        self.start_all_btn.setText(tr("start_all"))
        self.stop_all_btn.setText(tr("stop_all"))

        self.param_card.set_title(tr("global_params"))
        self.fps_lbl.setText(tr("fps_limit"))
        self.thresh_lbl.setText(tr("threshold"))
        self.sub_all_btn.setText(
            tr("sub_all_on") if self.sub_all_btn.isChecked() else tr("sub_all_off"))
        self.reset_all_bg_btn.setText(tr("reset_all_bg"))
        self.reset_all_bg_btn.setToolTip(tr("reset_all_bg_tip"))
        self.alert_lbl.setText(tr("alert_hdr"))
        self.global_alert_spin.setToolTip(tr("global_alert_tip"))
        self.apply_alert_btn.setText(tr("apply_all"))

        self.cam_param_card.set_title(tr("cam_params"))
        self.exp_key.setText(tr("exposure"))
        self.gain_key.setText(tr("gain"))
        self.fr_key.setText(tr("frame_rate"))
        self.get_param_btn.setText(tr("get_param"))
        self.set_param_btn.setText(tr("set_param"))

        self.dev_card.set_title(tr("detected_dev"))
        self.overview_btn.setText(tr("overview"))
        self.ov_title.setText(tr("overview_title"))
        self.back_btn.setText(tr("back_to_tabs"))

        for page in range(self.tabs.count()):
            first = page * self.per_page + 1
            last = min((page + 1) * self.per_page, self.max_cameras)
            self.tabs.setTabText(
                page, tr("page_tab", p=page + 1, a=first, b=last))

        for tile in self._tiles:
            tile.retranslate()
        self._update_stream_count()

    # ── shutdown ──────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        try:
            self.service.shutdown()
        except Exception as exc:
            log.warning("Shutdown error: %s", exc)
        event.accept()
