#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
single_cam_stream.py
====================
單相機實時監控應用 (V1.0)

功能清單:
  1. ✅ RGB、曝光的平均值顯示
  2. ✅ 調整白平衡
  3. ✅ 顯示 HSL
  4. ✅ 顯示這張圖的平均光強
  5. ✅ 調整畫面大小

架構:
  - modules/image_analyzer.py     → 影像分析
  - modules/white_balance.py      → 白平衡控制
  - modules/frame_resizer.py      → 畫面縮放
  - modules/camera_controller.py  → 相機控制
  - modules/ui_components.py      → UI 元件
  - single_cam_stream.py          → 主應用

使用:
  python single_cam_stream.py

作者: Robert Lee
版本: 1.0.0
"""

import sys
import os

# 加入模組路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import threading
import time
import cv2
import numpy as np
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QMessageBox, QScrollArea,
    QTabWidget, QStatusBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QIcon

from modules import (
    ImageAnalyzer,
    WhiteBalanceController,
    FrameResizer,
    CameraController,
    VideoLabel,
    StatusPanel,
    WhiteBalanceControl,
    FrameResizeControl
)


class FrameCapturerThread(QThread):
    """後台取幀執行緒"""
    
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, camera: CameraController):
        super().__init__()
        self.camera = camera
        self.running = True
    
    def run(self):
        """執行緒主循環"""
        while self.running:
            if self.camera.is_grabbing:
                frame = self.camera.get_frame(timeout_ms=50)
                if frame is not None:
                    self.frame_ready.emit(frame)
            else:
                time.sleep(0.01)
    
    def stop(self):
        """停止執行緒"""
        self.running = False
        self.wait()


class SingleCameraApp(QMainWindow):
    """單相機應用主視窗"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎥 單相機實時監控系統 V1.0")
        self.setGeometry(100, 100, 1400, 900)
        
        # 初始化核心模組
        self.camera = CameraController()
        self.analyzer = ImageAnalyzer()
        self.white_balance = WhiteBalanceController()
        self.resizer = FrameResizer()
        
        # UI 元件
        self.video_label = None
        self.status_panel = None
        self.wb_control = None
        self.resize_control = None
        
        # 執行緒
        self.frame_thread: Optional[FrameCapturerThread] = None
        self.timer: Optional[QTimer] = None
        
        # 狀態
        self.is_running = False
        self.current_frame = None
        
        # 建立 UI
        self._setup_ui()
        
        # 連接信號
        self._connect_signals()
    
    def _setup_ui(self):
        """設置用戶界面"""
        central = QWidget()
        main_layout = QVBoxLayout()
        
        # ==================== 頂部控制列 ====================
        toolbar = self._create_toolbar()
        main_layout.addLayout(toolbar)
        
        # ==================== 主顯示區域 ====================
        display_layout = QHBoxLayout()
        
        # 左側：影像顯示
        left_layout = QVBoxLayout()
        self.video_label = VideoLabel()
        left_layout.addWidget(self.video_label)
        
        # 右側：控制面板
        right_layout = QVBoxLayout()
        
        # 狀態面板
        self.status_panel = StatusPanel()
        right_layout.addWidget(self.status_panel)
        
        # 白平衡控制
        self.wb_control = WhiteBalanceControl()
        right_layout.addWidget(self.wb_control)
        
        # 畫面縮放控制
        self.resize_control = FrameResizeControl()
        right_layout.addWidget(self.resize_control)
        
        right_layout.addStretch()
        
        # 組合左右佈局
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        right_scroll = QScrollArea()
        right_scroll.setWidget(right_widget)
        right_scroll.setWidgetResizable(True)
        right_scroll.setFixedWidth(400)
        
        display_layout.addLayout(left_layout, 3)
        display_layout.addWidget(right_scroll, 1)
        main_layout.addLayout(display_layout)
        
        # ==================== 狀態欄 ====================
        self.statusBar().showMessage("就緒")
        
        central.setLayout(main_layout)
        self.setCentralWidget(central)
    
    def _create_toolbar(self) -> QHBoxLayout:
        """建立頂部工具欄"""
        toolbar = QHBoxLayout()
        
        # 相機選擇
        toolbar.addWidget(QLabel("相機:"))
        self.camera_combo = QComboBox()
        toolbar.addWidget(self.camera_combo)
        
        # 刷新設備按鈕
        refresh_btn = QPushButton("🔄 刷新設備")
        refresh_btn.setFixedHeight(32)
        refresh_btn.clicked.connect(self._on_refresh_devices)
        toolbar.addWidget(refresh_btn)
        
        # 連接按鈕
        self.connect_btn = QPushButton("✅ 連接")
        self.connect_btn.setFixedHeight(32)
        self.connect_btn.setStyleSheet("background-color: #5cb85c;")
        self.connect_btn.clicked.connect(self._on_connect)
        toolbar.addWidget(self.connect_btn)
        
        # 開始/停止按鈕
        self.start_btn = QPushButton("▶️  開始取像")
        self.start_btn.setFixedHeight(32)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("background-color: #0275d8;")
        self.start_btn.clicked.connect(self._on_start_stop)
        toolbar.addWidget(self.start_btn)
        
        toolbar.addStretch()
        
        # 退出按鈕
        exit_btn = QPushButton("❌ 退出")
        exit_btn.setFixedHeight(32)
        exit_btn.setStyleSheet("background-color: #d9534f;")
        exit_btn.clicked.connect(self.close)
        toolbar.addWidget(exit_btn)
        
        return toolbar
    
    def _connect_signals(self):
        """連接信號槽"""
        # 白平衡信號
        self.wb_control.gains_changed.connect(self._on_gains_changed)
        self.wb_control.preset_selected.connect(self._on_preset_selected)
        
        # 縮放信號
        self.resize_control.scale_changed.connect(self._on_scale_changed)
        self.resize_control.size_changed.connect(self._on_size_changed)
        
        # 更新計時器
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_update_frame)
    
    def _on_refresh_devices(self):
        """刷新設備列表"""
        self.camera_combo.clear()
        devices = self.camera.enum_devices()
        
        if not devices:
            self.statusBar().showMessage("❌ 未找到相機設備")
            QMessageBox.warning(self, "錯誤", "未找到相機設備")
            return
        
        for dev in devices:
            self.camera_combo.addItem(f"Camera {dev['index']} ({dev['ip']})", dev['index'])
        
        self.statusBar().showMessage(f"✅ 找到 {len(devices)} 台相機")
    
    def _on_connect(self):
        """連接到相機"""
        if self.camera.is_connected:
            self.camera.disconnect()
            self.connect_btn.setText("✅ 連接")
            self.connect_btn.setStyleSheet("background-color: #5cb85c;")
            self.start_btn.setEnabled(False)
            self.statusBar().showMessage("已斷開連接")
            return
        
        device_index = self.camera_combo.currentData()
        if device_index is None:
            QMessageBox.warning(self, "錯誤", "請先刷新設備列表")
            return
        
        if self.camera.connect(device_index):
            self.connect_btn.setText("❌ 斷開")
            self.connect_btn.setStyleSheet("background-color: #d9534f;")
            self.start_btn.setEnabled(True)
            self.statusBar().showMessage(f"✅ 已連接到相機 #{device_index}")
        else:
            QMessageBox.critical(self, "錯誤", "連接失敗")
    
    def _on_start_stop(self):
        """開始/停止取像"""
        if self.is_running:
            self._stop_grabbing()
        else:
            self._start_grabbing()
    
    def _start_grabbing(self):
        """開始取像"""
        if not self.camera.is_connected:
            QMessageBox.warning(self, "錯誤", "請先連接相機")
            return
        
        if not self.camera.start_grabbing():
            QMessageBox.critical(self, "錯誤", "啟動取像失敗")
            return
        
        # 啟動後台執行緒
        self.frame_thread = FrameCapturerThread(self.camera)
        self.frame_thread.frame_ready.connect(self._on_frame_received)
        self.frame_thread.start()
        
        # 啟動更新計時器
        self.timer.start(30)  # 30ms 更新一次 (~33 FPS)
        
        self.is_running = True
        self.start_btn.setText("⏹️  停止取像")
        self.start_btn.setStyleSheet("background-color: #f0ad4e;")
        self.camera_combo.setEnabled(False)
        self.connect_btn.setEnabled(False)
        self.statusBar().showMessage("▶️  正在取像中...")
    
    def _stop_grabbing(self):
        """停止取像"""
        self.timer.stop()
        self.camera.stop_grabbing()
        
        if self.frame_thread:
            self.frame_thread.stop()
        
        self.is_running = False
        self.start_btn.setText("▶️  開始取像")
        self.start_btn.setStyleSheet("background-color: #0275d8;")
        self.camera_combo.setEnabled(True)
        self.connect_btn.setEnabled(True)
        self.statusBar().showMessage("已停止取像")
    
    def _on_frame_received(self, frame: np.ndarray):
        """接收幀數據"""
        self.current_frame = frame.copy()
    
    def _on_update_frame(self):
        """更新顯示幀"""
        if self.current_frame is None:
            return
        
        frame = self.current_frame.copy()
        
        # 1. 應用白平衡
        frame = self.white_balance.apply_white_balance(frame)
        
        # 2. 應用縮放
        frame = self.resizer.resize(frame)
        
        # 3. 分析影像
        stats = self.analyzer.analyze_frame(frame)
        
        # 4. 更新顯示
        self.video_label.display_frame(frame)
        self.status_panel.update_stats(stats)
    
    def _on_gains_changed(self, r: float, g: float, b: float):
        """白平衡增益改變"""
        self.white_balance.set_gains(r, g, b)
    
    def _on_preset_selected(self, preset: str):
        """白平衡預設選擇"""
        if self.white_balance.set_preset(preset):
            r, g, b = self.white_balance.get_current_gains()
            self.wb_control.set_gains(r, g, b)
    
    def _on_scale_changed(self, ratio: float):
        """縮放比例改變"""
        self.resizer.set_scale_ratio(ratio)
    
    def _on_size_changed(self, width: int, height: int):
        """自訂尺寸改變"""
        if width > 0:
            self.resizer.set_target_width(width)
    
    def closeEvent(self, event):
        """關閉事件"""
        if self.is_running:
            self._stop_grabbing()
        
        if self.camera.is_connected:
            self.camera.disconnect()
        
        event.accept()


def main():
    """主函數"""
    app = QApplication(sys.argv)
    
    # 設置全局字體
    font = QFont("微軟正黑體" if sys.platform == "win32" else "Ubuntu", 10)
    app.setFont(font)
    
    # 建立並顯示主視窗
    window = SingleCameraApp()
    window.show()
    
    # 自動刷新設備
    window._on_refresh_devices()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
