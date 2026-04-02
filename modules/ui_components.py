# -*- coding: utf-8 -*-
"""
ui_components.py
================
可復用的 UI 元件 - 用於單相機應用

提供的元件:
  - 影像顯示標籤
  - 狀態資訊面板
  - 控制滑塊組
  - 預設按鈕組
"""

from PyQt5.QtWidgets import (
    QLabel, QSlider, QPushButton, QGroupBox, QVBoxLayout, QHBoxLayout,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
import numpy as np
import cv2


class VideoLabel(QLabel):
    """影像顯示標籤"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("border: 2px solid #333; background-color: #000;")
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
    
    def display_frame(self, frame: np.ndarray) -> None:
        """
        顯示 numpy 影像陣列
        
        Args:
            frame: BGR 格式影像
        """
        if frame is None or frame.size == 0:
            return
        
        # BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 轉換為 QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = 3 * w
        qt_image = QImage(
            rgb_frame.data,
            w, h,
            bytes_per_line,
            QImage.Format_RGB888
        )
        
        # 縮放到標籤尺寸
        pixmap = QPixmap.fromImage(qt_image)
        pixmap = pixmap.scaledToWidth(
            self.width(),
            Qt.SmoothTransformation
        )
        
        self.setPixmap(pixmap)


class StatusPanel(QGroupBox):
    """狀態資訊面板"""
    
    def __init__(self, parent=None):
        super().__init__("📊 影像統計", parent)
        
        # RGB 顯示
        self.rgb_label = QLabel("RGB: R=0 | G=0 | B=0")
        self.rgb_label.setFont(QFont("Courier", 10))
        
        # HSL 顯示
        self.hsl_label = QLabel("HSL: H=0° | S=0% | L=0%")
        self.hsl_label.setFont(QFont("Courier", 10))
        
        # 光強度顯示
        self.brightness_label = QLabel("光強: 128 (50.2%)")
        self.brightness_label.setFont(QFont("Courier", 10))
        
        # 曝光顯示
        self.exposure_label = QLabel("曝光: 5000 μs")
        self.exposure_label.setFont(QFont("Courier", 10))
        
        # 佈局
        layout = QVBoxLayout()
        layout.addWidget(self.rgb_label)
        layout.addWidget(self.hsl_label)
        layout.addWidget(self.brightness_label)
        layout.addWidget(self.exposure_label)
        self.setLayout(layout)
    
    def update_stats(self, stats: dict) -> None:
        """更新統計資訊"""
        if 'rgb_avg' in stats:
            r, g, b = stats['rgb_avg']
            self.rgb_label.setText(f"RGB: R={r:.1f} | G={g:.1f} | B={b:.1f}")
        
        if 'hsl_avg' in stats:
            h, s, l = stats['hsl_avg']
            self.hsl_label.setText(f"HSL: H={h:.1f}° | S={s:.1f}% | L={l:.1f}%")
        
        if 'brightness' in stats:
            brightness = stats['brightness']
            percent = (brightness / 255.0) * 100.0
            self.brightness_label.setText(f"光強: {brightness:.1f} ({percent:.1f}%)")
        
        if 'exposure_time' in stats:
            exp = stats['exposure_time']
            self.exposure_label.setText(f"曝光: {exp:.0f} μs")


class WhiteBalanceControl(QGroupBox):
    """白平衡控制面板"""
    
    gains_changed = pyqtSignal(float, float, float)  # r, g, b
    preset_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__("🎨 白平衡調整", parent)
        
        # R 增益滑塊
        r_layout = QHBoxLayout()
        r_layout.addWidget(QLabel("紅 (R):"))
        self.r_slider = QSlider(Qt.Horizontal)
        self.r_slider.setRange(50, 200)
        self.r_slider.setValue(100)
        self.r_slider.setTickInterval(10)
        self.r_slider.setTickPosition(QSlider.TicksBelow)
        self.r_value = QLabel("1.00")
        self.r_value.setFixedWidth(50)
        r_layout.addWidget(self.r_slider)
        r_layout.addWidget(self.r_value)
        
        # G 增益滑塊
        g_layout = QHBoxLayout()
        g_layout.addWidget(QLabel("綠 (G):"))
        self.g_slider = QSlider(Qt.Horizontal)
        self.g_slider.setRange(50, 200)
        self.g_slider.setValue(100)
        self.g_slider.setTickInterval(10)
        self.g_slider.setTickPosition(QSlider.TicksBelow)
        self.g_value = QLabel("1.00")
        self.g_value.setFixedWidth(50)
        g_layout.addWidget(self.g_slider)
        g_layout.addWidget(self.g_value)
        
        # B 增益滑塊
        b_layout = QHBoxLayout()
        b_layout.addWidget(QLabel("藍 (B):"))
        self.b_slider = QSlider(Qt.Horizontal)
        self.b_slider.setRange(50, 200)
        self.b_slider.setValue(100)
        self.b_slider.setTickInterval(10)
        self.b_slider.setTickPosition(QSlider.TicksBelow)
        self.b_value = QLabel("1.00")
        self.b_value.setFixedWidth(50)
        b_layout.addWidget(self.b_slider)
        b_layout.addWidget(self.b_value)
        
        # 預設按鈕
        preset_layout = QHBoxLayout()
        for preset in ['Default', 'Daylight', 'Cloudy', 'Tungsten', 'Fluorescent']:
            btn = QPushButton(preset)
            btn.setFixedHeight(28)
            btn.clicked.connect(lambda checked, p=preset: self.preset_selected.emit(p.lower()))
            preset_layout.addWidget(btn)
        
        # 重置按鈕
        reset_btn = QPushButton("重置")
        reset_btn.setFixedHeight(28)
        reset_btn.setStyleSheet("background-color: #f0ad4e;")
        reset_btn.clicked.connect(self._on_reset)
        preset_layout.addWidget(reset_btn)
        
        # 連接信號
        self.r_slider.valueChanged.connect(self._update_display)
        self.g_slider.valueChanged.connect(self._update_display)
        self.b_slider.valueChanged.connect(self._update_display)
        
        # 佈局
        main_layout = QVBoxLayout()
        main_layout.addLayout(r_layout)
        main_layout.addLayout(g_layout)
        main_layout.addLayout(b_layout)
        main_layout.addLayout(preset_layout)
        self.setLayout(main_layout)
    
    def _update_display(self):
        """更新滑塊值顯示"""
        r = self.r_slider.value() / 100.0
        g = self.g_slider.value() / 100.0
        b = self.b_slider.value() / 100.0
        
        self.r_value.setText(f"{r:.2f}")
        self.g_value.setText(f"{g:.2f}")
        self.b_value.setText(f"{b:.2f}")
        
        self.gains_changed.emit(r, g, b)
    
    def _on_reset(self):
        """重置為預設值"""
        self.r_slider.setValue(100)
        self.g_slider.setValue(100)
        self.b_slider.setValue(100)
    
    def set_gains(self, r: float, g: float, b: float) -> None:
        """設定增益"""
        self.r_slider.blockSignals(True)
        self.g_slider.blockSignals(True)
        self.b_slider.blockSignals(True)
        
        self.r_slider.setValue(int(r * 100))
        self.g_slider.setValue(int(g * 100))
        self.b_slider.setValue(int(b * 100))
        
        self.r_slider.blockSignals(False)
        self.g_slider.blockSignals(False)
        self.b_slider.blockSignals(False)
        
        self._update_display()


class FrameResizeControl(QGroupBox):
    """畫面縮放控制面板"""
    
    scale_changed = pyqtSignal(float)
    size_changed = pyqtSignal(int, int)
    
    def __init__(self, parent=None):
        super().__init__("📐 畫面縮放", parent)
        
        # 預設按鈕
        preset_layout = QHBoxLayout()
        for preset in ['100%', '75%', '50%', '25%']:
            btn = QPushButton(preset)
            btn.setFixedHeight(28)
            btn.clicked.connect(lambda checked, p=preset: self._on_preset(p))
            preset_layout.addWidget(btn)
        
        # 自訂寬度
        custom_layout = QHBoxLayout()
        custom_layout.addWidget(QLabel("自訂寬度:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(100, 3840)
        self.width_spin.setValue(640)
        self.width_spin.setSuffix(" px")
        self.width_spin.valueChanged.connect(lambda v: self.size_changed.emit(v, 0))
        custom_layout.addWidget(self.width_spin)
        
        # 佈局
        main_layout = QVBoxLayout()
        main_layout.addLayout(preset_layout)
        main_layout.addLayout(custom_layout)
        self.setLayout(main_layout)
    
    def _on_preset(self, preset: str):
        """預設按鈕點擊"""
        ratio = {
            '100%': 1.0,
            '75%': 0.75,
            '50%': 0.5,
            '25%': 0.25,
        }
        self.scale_changed.emit(ratio.get(preset, 1.0))


# 測試用例
if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
    
    app = QApplication([])
    
    window = QMainWindow()
    window.setWindowTitle("UI 元件測試")
    
    central = QWidget()
    layout = QVBoxLayout()
    
    # 測試狀態面板
    status = StatusPanel()
    status.update_stats({
        'rgb_avg': (100, 150, 200),
        'hsl_avg': (180, 50, 75),
        'brightness': 150,
        'exposure_time': 5000
    })
    layout.addWidget(status)
    
    # 測試白平衡控制
    wb = WhiteBalanceControl()
    layout.addWidget(wb)
    
    # 測試縮放控制
    resize = FrameResizeControl()
    layout.addWidget(resize)
    
    central.setLayout(layout)
    window.setCentralWidget(central)
    window.show()
    
    app.exec_()
