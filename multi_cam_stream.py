"""
multi_cam_stream.py
====================
V4.0.0 重新佈局：
  - 每頁 2 台相機（左右並排），共 10 頁，支援最多 20 台 POE 相機
  - 字體全面放大（13px 基礎，工具列 14px）
  - 工具列高度 32px，按鈕清晰易點按
  - 雙擊影像區可全螢幕預覽單台相機
  - 右上角 PiP 顯示影像相減結果
  - 全域硬體監控背景執行緒（不阻塞 UI）
  - CPU / CUDA 雙模式運算引擎
  - 自動偵測 GigE / USB3 Vision 相機並顯示型號+IP

作者：Robert Lee  版本：V4.0.0
"""

import sys
import threading
import numpy as np
import time
import os
import psutil
import subprocess
import platform
from ctypes import *


# ===============================================
# 支援 Windows / Linux 跨平台的 MVS SDK 路徑匯入
# ===============================================
if platform.system() == "Windows":
    mvs_path = "C:/Program Files (x86)/MVS/Development/Samples/Python/MvImport"
    if os.path.exists(mvs_path):
        sys.path.append(mvs_path)
else:
    linux_paths = [
        "/opt/MVS/Samples/aarch64/Python/MvImport",
        "/opt/MVS/Samples/x86_64/Python/MvImport",
        "/opt/MVS/Samples/64/Python/MvImport"
    ]
    for p in linux_paths:
        if os.path.exists(p) and p not in sys.path:
            sys.path.append(p)

try:
    from MvCameraControl_class import *
    from CameraParams_header import *
except ImportError:
    print("【嚴重錯誤】: 找不到海康威視 MVS Python SDK，請確認 MVS 已正確安裝。")

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QComboBox, QSpinBox, QGroupBox,
    QMessageBox, QRadioButton, QButtonGroup, QScrollArea, QGridLayout,
    QCheckBox, QSizePolicy, QFrame, QSplitter, QListWidget, QListWidgetItem,
    QTabWidget, QDoubleSpinBox, QDialog
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QCursor

import cv2

# ── 佈局常數 ─────────────────────────────────────────────────
MAX_CAMERAS   = 20
CAMS_PER_PAGE = 2    # 每頁 2 台，左右並排
PAGE_COUNT    = MAX_CAMERAS // CAMS_PER_PAGE   # 10 頁


# -------------------------------------------------------
# 工具函式：判斷 CUDA 是否可用
# -------------------------------------------------------
def is_cuda_supported() -> bool:
    try:
        return hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False

HAS_CUDA = is_cuda_supported()


# ===============================================================
# MonitorThread：硬體狀態監控執行緒
#   - 定期背景讀取 CPU/RAM/GPU，避免阻塞主 UI 執行緒
# ===============================================================
class MonitorThread(QThread):
    stats_ready = pyqtSignal(float, float, str, str)  # cpu%, ram%, gpu_util, vram

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.has_nvidia = True

    def run(self):
        while self.running:
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            gpu_util, vram = "--", "-- / -- MB"

            if self.has_nvidia:
                try:
                    kwargs = {}
                    if platform.system() == "Windows":
                        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                    res = subprocess.check_output(
                        ["nvidia-smi",
                         "--query-gpu=utilization.gpu,memory.used,memory.total",
                         "--format=csv,noheader,nounits"],
                        **kwargs
                    ).decode().strip().split(",")
                    if len(res) == 3:
                        u, mu, mt = res
                        gpu_util = u.strip()
                        vram = f"{mu.strip()} / {mt.strip()} MB"
                except Exception:
                    self.has_nvidia = False

            self.stats_ready.emit(cpu, ram, gpu_util, vram)
            time.sleep(1.5)

    def stop(self):
        self.running = False
        self.wait()


# ===============================================================
# CameraThread：單一相機取像執行緒
# ===============================================================
class CameraThread(QThread):
    frame_ready = pyqtSignal(int, np.ndarray, np.ndarray, float)

    def __init__(self, cam_obj, cam_id: int, use_cuda: bool = False, parent=None):
        super().__init__(parent)
        self.cam     = cam_obj
        self.cam_id  = cam_id
        self.use_cuda = use_cuda

        self.running = True
        self.target_fps     = 15
        self.diff_threshold = 30

        self._subtraction_enabled = False
        self._lock = threading.Lock()
        self.bg_frame = None

    def set_subtraction(self, enabled: bool):
        with self._lock:
            self._subtraction_enabled = enabled
        if not enabled:
            self.bg_frame = None

    def get_subtraction(self) -> bool:
        with self._lock:
            return self._subtraction_enabled

    def run(self):
        stFrameInfo   = MV_FRAME_OUT_INFO_EX()
        stPayloadSize = MVCC_INTVALUE_EX()

        ret = self.cam.MV_CC_GetIntValueEx("PayloadSize", stPayloadSize)
        if ret != 0:
            print(f"[Cam {self.cam_id}] 無法獲取 PayloadSize")
            return

        nPayloadSize = stPayloadSize.nCurValue
        pData = (c_ubyte * nPayloadSize)()

        # 預先配置 RGB 轉換記憶體（Payload * 3 = 足夠 BGR8）
        max_rgb_size = nPayloadSize * 3
        pDataForRGB  = (c_ubyte * max_rgb_size)()

        if self.use_cuda:
            gpu_frame       = cv2.cuda_GpuMat()
            gpu_bg_f32      = cv2.cuda_GpuMat()
            gpu_gray_f32    = cv2.cuda_GpuMat()
            gpu_bg_8u       = cv2.cuda_GpuMat()
            gaussian_filter = cv2.cuda.createGaussianFilter(
                cv2.CV_8UC1, cv2.CV_8UC1, (21, 21), 0)
            is_bg_initialized = False

        while self.running:
            t0 = time.time()

            ret = self.cam.MV_CC_GetOneFrameTimeout(pData, nPayloadSize, stFrameInfo)
            if ret == 0:
                stConvert = MV_CC_PIXEL_CONVERT_PARAM()
                stConvert.nWidth          = stFrameInfo.nWidth
                stConvert.nHeight         = stFrameInfo.nHeight
                stConvert.pSrcData        = cast(pData, POINTER(c_ubyte))
                stConvert.nSrcDataLen     = stFrameInfo.nFrameLen
                stConvert.enSrcPixelType  = stFrameInfo.enPixelType

                nRGBSize = stFrameInfo.nWidth * stFrameInfo.nHeight * 3
                if nRGBSize > max_rgb_size:
                    max_rgb_size = nRGBSize
                    pDataForRGB  = (c_ubyte * max_rgb_size)()

                stConvert.enDstPixelType = PixelType_Gvsp_BGR8_Packed
                stConvert.pDstBuffer     = cast(pDataForRGB, POINTER(c_ubyte))
                stConvert.nDstBufferSize = max_rgb_size

                if self.cam.MV_CC_ConvertPixelType(stConvert) != 0:
                    continue

                temp_arr  = np.frombuffer(pDataForRGB, count=nRGBSize, dtype=np.uint8)
                frame_bgr = temp_arr.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth, 3))
                frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

                do_sub = self.get_subtraction()
                coverage_pct = 0.0

                if do_sub:
                    if self.use_cuda:
                        gpu_frame.upload(frame_gray)
                        gpu_gray = gaussian_filter.apply(gpu_frame)
                        gpu_gray.convertTo(cv2.CV_32F, gpu_gray_f32)
                        if not is_bg_initialized:
                            gpu_gray.convertTo(cv2.CV_32F, gpu_bg_f32)
                            is_bg_initialized = True
                        gpu_bg_f32 = cv2.cuda.addWeighted(
                            gpu_gray_f32, 0.01, gpu_bg_f32, 0.99, 0.0)
                        gpu_bg_f32.convertTo(cv2.CV_8U, gpu_bg_8u)
                        gpu_diff = cv2.cuda.absdiff(gpu_bg_8u, gpu_frame)
                        _, gpu_thresh = cv2.cuda.threshold(
                            gpu_diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
                        thresh_res = gpu_thresh.download()
                    else:
                        gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
                        if self.bg_frame is None or self.bg_frame.shape != gray.shape:
                            self.bg_frame = gray.copy().astype("float")
                        cv2.accumulateWeighted(gray, self.bg_frame, 0.01)
                        bg_current = cv2.convertScaleAbs(self.bg_frame)
                        frame_diff = cv2.absdiff(bg_current, gray)
                        _, thresh_res = cv2.threshold(
                            frame_diff, self.diff_threshold, 255, cv2.THRESH_BINARY)

                    # ── Contour 覆蓋率計算 ─────────────────────
                    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    cleaned = cv2.morphologyEx(thresh_res, cv2.MORPH_CLOSE, kernel)
                    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel)

                    contours, _ = cv2.findContours(
                        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    total_pixels   = thresh_res.shape[0] * thresh_res.shape[1]
                    covered_pixels = sum(cv2.contourArea(c) for c in contours)
                    coverage_pct   = (covered_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

                    diff_display = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
                    cv2.drawContours(diff_display, contours, -1, (0, 255, 100), 2)
                    cv2.putText(diff_display, f"Coverage: {coverage_pct:.1f}%",
                                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2)
                else:
                    diff_display = np.zeros_like(frame_bgr)
                    cv2.putText(diff_display, "Subtraction OFF",
                                (10, frame_bgr.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)

                self.frame_ready.emit(
                    self.cam_id, frame_bgr.copy(), diff_display.copy(), coverage_pct)

            elapsed = time.time() - t0
            sleep_t = max(1.0 / self.target_fps - elapsed, 0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    def stop(self):
        self.running = False
        self.wait()


# ===============================================================
# FullscreenDialog：雙擊影像後的全螢幕獨立視窗
# ===============================================================
class FullscreenDialog(QDialog):
    def __init__(self, title: str, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle(f"全螢幕 — {title}")
        self.setStyleSheet("background:#000;")
        self.showMaximized()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.img_lbl = QLabel()
        self.img_lbl.setAlignment(Qt.AlignCenter)
        self.img_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img_lbl.setText("等待影像...")
        self.img_lbl.setStyleSheet("color:#555; font-size:20px;")
        layout.addWidget(self.img_lbl)

        hint = QLabel("按 Esc 或雙擊關閉")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color:#333; font-size:12px; padding: 4px;")
        layout.addWidget(hint)

    def update_frame(self, bgr: np.ndarray):
        w = self.img_lbl.width()
        h = self.img_lbl.height()
        if w > 0 and h > 0:
            h_img, w_img, ch = bgr.shape
            rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            qimg  = QImage(rgb.data, w_img, h_img, ch * w_img, QImage.Format_RGB888)
            pix   = QPixmap.fromImage(qimg.copy()).scaled(
                w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img_lbl.setPixmap(pix)

    def mouseDoubleClickEvent(self, event):
        self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        super().keyPressEvent(event)


# ===============================================================
# CameraWidget：單一相機的顯示槽位 V4
#   佈局：
#     ┌── 工具列 (32px) ─────────────────────────────────┐
#     │ #01  [下拉選單──────────────]  Cov:0.0%  [相減] [●]│
#     ├──────────────────────────────────────────────────┤
#     │          主影像（全寬全高，雙擊全螢幕）              │
#     │                               ┌── PiP 差異 ──┐  │
#     │                               └──────────────┘  │
#     └──────────────────────────────────────────────────┘
# ===============================================================
class CameraWidget(QFrame):
    subtraction_toggled = pyqtSignal(int, bool)   # (cam_id, enabled)
    slot_cam_changed    = pyqtSignal(int, int)     # (slot_id, cam_id)

    def __init__(self, slot_id: int, cam_labels: list, parent=None):
        super().__init__(parent)
        self.slot_id   = slot_id
        self.cam_labels = cam_labels
        self._current_cam_id = slot_id
        self._sub_enabled = False
        self._coverage_alert_threshold = 50.0
        self._alert_active = False
        self._fullscreen_dlg: FullscreenDialog | None = None
        self._last_bgr: np.ndarray | None = None
        self._setup_ui()

    # ─────────────────────────────────────────────────────
    #  UI 建構
    # ─────────────────────────────────────────────────────
    def _setup_ui(self):
        self.setFrameShape(QFrame.Box)
        self.setLineWidth(1)
        self._apply_normal_style()

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(4)

        # ── 工具列 ─────────────────────────────────────────
        bar = QHBoxLayout()
        bar.setSpacing(8)
        bar.setContentsMargins(0, 0, 0, 0)

        # 槽位編號標籤
        slot_lbl = QLabel(f"#{self.slot_id + 1:02d}")
        slot_lbl.setFixedWidth(30)
        slot_lbl.setStyleSheet(
            "color:#7799dd; font-weight:bold; font-size:14px;")
        bar.addWidget(slot_lbl)

        # 相機選擇下拉
        self.cam_combo = QComboBox()
        self.cam_combo.setMinimumHeight(30)
        self.cam_combo.setStyleSheet("""
            QComboBox {
                background:#1a1a30; border:1px solid #3a3a5a;
                border-radius:4px; padding:2px 6px;
                font-size:13px; color:#a8c0ff;
                min-width:180px;
            }
            QComboBox::drop-down { border:none; width:18px; }
            QComboBox QAbstractItemView {
                background:#1a1a2e; selection-background-color:#2a2a5a;
                font-size:13px;
            }
        """)
        self.cam_combo.addItem("─ 未指派 ─", -1)
        for i, label in enumerate(self.cam_labels):
            self.cam_combo.addItem(label, i)
        if self.slot_id < len(self.cam_labels):
            self.cam_combo.setCurrentIndex(self.slot_id + 1)
        self.cam_combo.currentIndexChanged.connect(self._on_combo_changed)
        bar.addWidget(self.cam_combo, 1)

        # 解析度標籤
        self.res_lbl = QLabel("----×----")
        self.res_lbl.setFixedWidth(90)
        self.res_lbl.setStyleSheet(
            "color:#668899; font-size:12px; qproperty-alignment:AlignRight;")
        bar.addWidget(self.res_lbl)

        # 覆蓋率標籤
        cov_icon = QLabel("▣")
        cov_icon.setStyleSheet("color:#5588bb; font-size:14px;")
        bar.addWidget(cov_icon)
        self.cov_lbl = QLabel("0.0%")
        self.cov_lbl.setFixedWidth(52)
        self.cov_lbl.setStyleSheet(
            "color:#55ccee; font-size:13px; font-weight:bold;")
        bar.addWidget(self.cov_lbl)

        # Alert 閾值
        alert_icon = QLabel("⚠")
        alert_icon.setStyleSheet("color:#cc9922; font-size:14px;")
        bar.addWidget(alert_icon)
        self.alert_spin = QDoubleSpinBox()
        self.alert_spin.setRange(0.0, 100.0)
        self.alert_spin.setValue(50.0)
        self.alert_spin.setSuffix("%")
        self.alert_spin.setDecimals(0)
        self.alert_spin.setFixedWidth(72)
        self.alert_spin.setMinimumHeight(30)
        self.alert_spin.setToolTip("覆蓋率超過此值觸發 Alert")
        self.alert_spin.setStyleSheet("""
            QDoubleSpinBox {
                background:#1a1a30; border:1px solid #3a3a5a;
                border-radius:4px; padding:2px; font-size:13px;
            }
        """)
        self.alert_spin.valueChanged.connect(self._on_alert_threshold_changed)
        bar.addWidget(self.alert_spin)

        self.alert_lbl = QLabel()
        self.alert_lbl.setFixedWidth(64)
        self.alert_lbl.setStyleSheet(
            "color:#ff5555; font-size:12px; font-weight:bold;")
        bar.addWidget(self.alert_lbl)

        # 相減開關按鈕
        self.sub_btn = QPushButton("影像相減")
        self.sub_btn.setCheckable(True)
        self.sub_btn.setFixedHeight(30)
        self.sub_btn.setMinimumWidth(80)
        self.sub_btn.setStyleSheet("""
            QPushButton {
                background:#222238; color:#7788aa; border-radius:5px;
                font-size:13px; border:1px solid #3a3a55;
                padding:0 8px;
            }
            QPushButton:checked {
                background:#1a5a32; color:#80ffaa; border:1px solid #2ecc71;
            }
        """)
        self.sub_btn.clicked.connect(self._on_sub_toggled)
        bar.addWidget(self.sub_btn)

        # 狀態燈
        self.status_lbl = QLabel("●")
        self.status_lbl.setFixedWidth(18)
        self.status_lbl.setStyleSheet("color:#3a3a55; font-size:16px;")
        self.status_lbl.setToolTip("未連線")
        bar.addWidget(self.status_lbl)

        root.addLayout(bar)

        # ── 影像區 ──────────────────────────────────────────
        self.img_container = QWidget()
        self.img_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img_container.setMinimumHeight(120)
        self.img_container.setStyleSheet("background:#060610; border-radius:4px;")

        # 主影像
        self.orig_lbl = QLabel(self.img_container)
        self.orig_lbl.setAlignment(Qt.AlignCenter)
        self.orig_lbl.setStyleSheet(
            "background:transparent; color:#2a2a4a; font-size:14px;")
        self.orig_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.orig_lbl.setText("等待連線...")

        # PiP 差異縮圖
        self.diff_lbl = QLabel(self.img_container)
        self.diff_lbl.setAlignment(Qt.AlignCenter)
        self.diff_lbl.setStyleSheet("""
            background:#0a1020;
            border:1px solid #2a3a5a;
            border-radius:3px;
            color:#333; font-size:11px;
        """)
        self.diff_lbl.setText("差異")
        self.diff_lbl.setVisible(False)

        root.addWidget(self.img_container, 1)

    # ─────────────────────────────────────────────────────
    #  樣式
    # ─────────────────────────────────────────────────────
    def _apply_normal_style(self):
        self.setStyleSheet("""
            CameraWidget {
                border: 1px solid #2e2e4a;
                border-radius: 7px;
                background: #12122a;
            }
        """)

    def _apply_alert_style(self):
        self.setStyleSheet("""
            CameraWidget {
                border: 2px solid #ff4444;
                border-radius: 7px;
                background: #1e0808;
            }
        """)

    # ─────────────────────────────────────────────────────
    #  槽位對應的相機 ID
    # ─────────────────────────────────────────────────────
    def current_cam_id(self) -> int:
        return self._current_cam_id

    def _on_combo_changed(self, index: int):
        cam_id = self.cam_combo.itemData(index)
        self._current_cam_id = cam_id if cam_id is not None else -1
        self.slot_cam_changed.emit(self.slot_id, self._current_cam_id)

    def set_cam_labels(self, cam_labels: list):
        old_cam_id = self._current_cam_id
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        self.cam_combo.addItem("─ 未指派 ─", -1)
        for i, label in enumerate(cam_labels):
            self.cam_combo.addItem(label, i)
        for i in range(self.cam_combo.count()):
            if self.cam_combo.itemData(i) == old_cam_id:
                self.cam_combo.setCurrentIndex(i)
                break
        self.cam_combo.blockSignals(False)

    # ─────────────────────────────────────────────────────
    #  子 Widget 位置隨容器縮放
    # ─────────────────────────────────────────────────────
    def resizeEvent(self, event):
        super().resizeEvent(event)
        cw = self.img_container.width()
        ch = self.img_container.height()
        # 主畫面滿版
        self.orig_lbl.setGeometry(0, 0, cw, ch)
        # PiP：右上角，寬 28%，高按長寬比估算，最小 100×75
        pip_w  = max(int(cw * 0.28), 100)
        pip_h  = max(int(pip_w * 3 / 4), 75)
        margin = 6
        self.diff_lbl.setGeometry(cw - pip_w - margin, margin, pip_w, pip_h)

    # ─────────────────────────────────────────────────────
    #  雙擊 → 全螢幕預覽
    # ─────────────────────────────────────────────────────
    def mouseDoubleClickEvent(self, event):
        if self._fullscreen_dlg is None or not self._fullscreen_dlg.isVisible():
            title = self.cam_combo.currentText()
            self._fullscreen_dlg = FullscreenDialog(title, self)
            if self._last_bgr is not None:
                self._fullscreen_dlg.update_frame(self._last_bgr)
            self._fullscreen_dlg.show()
        super().mouseDoubleClickEvent(event)

    # ─────────────────────────────────────────────────────
    #  相減按鈕
    # ─────────────────────────────────────────────────────
    def _on_sub_toggled(self, checked: bool):
        self._sub_enabled = checked
        self.sub_btn.setText("相減 ON" if checked else "影像相減")
        self.diff_lbl.setVisible(checked)
        self.subtraction_toggled.emit(self._current_cam_id, checked)

    # ─────────────────────────────────────────────────────
    #  Alert 閾值
    # ─────────────────────────────────────────────────────
    def _on_alert_threshold_changed(self, val: float):
        self._coverage_alert_threshold = val

    def set_alert_threshold(self, val: float):
        self.alert_spin.blockSignals(True)
        self.alert_spin.setValue(val)
        self._coverage_alert_threshold = val
        self.alert_spin.blockSignals(False)

    # ─────────────────────────────────────────────────────
    #  影像更新
    # ─────────────────────────────────────────────────────
    def update_frames(self, bgr: np.ndarray, diff: np.ndarray, coverage: float):
        self._last_bgr = bgr

        # 主畫面
        w = self.orig_lbl.width()
        h = self.orig_lbl.height()
        if w > 0 and h > 0:
            self.orig_lbl.setPixmap(self._to_pixmap(bgr, w, h))

        # 同步更新全螢幕對話框
        if self._fullscreen_dlg is not None and self._fullscreen_dlg.isVisible():
            self._fullscreen_dlg.update_frame(bgr)

        # PiP 差異
        if self._sub_enabled:
            pw = self.diff_lbl.width()
            ph = self.diff_lbl.height()
            if pw > 0 and ph > 0:
                self.diff_lbl.setPixmap(self._to_pixmap(diff, pw, ph))

        # 覆蓋率文字
        self.cov_lbl.setText(f"{coverage:.1f}%")

        # 解析度
        h_img, w_img = bgr.shape[:2]
        self.res_lbl.setText(f"{w_img}×{h_img}")

        # Alert 狀態
        if coverage >= self._coverage_alert_threshold:
            if not self._alert_active:
                self._alert_active = True
                self._apply_alert_style()
            self.alert_lbl.setText(f"⚠ >{self._coverage_alert_threshold:.0f}%")
        else:
            if self._alert_active:
                self._alert_active = False
                self._apply_normal_style()
            self.alert_lbl.setText("")

        # 狀態燈
        self.status_lbl.setText("●")
        self.status_lbl.setStyleSheet("color:#2ecc71; font-size:16px;")
        self.status_lbl.setToolTip("串流中")

    def set_disconnected(self):
        self._last_bgr = None
        self.orig_lbl.setText("已停止")
        self.orig_lbl.setPixmap(QPixmap())
        self.diff_lbl.setVisible(False)
        self.res_lbl.setText("----×----")
        self.status_lbl.setText("●")
        self.status_lbl.setStyleSheet("color:#cc3333; font-size:16px;")
        self.status_lbl.setToolTip("已停止")
        self.cov_lbl.setText("--")
        self.alert_lbl.setText("")
        self._alert_active = False
        self._apply_normal_style()

    @staticmethod
    def _to_pixmap(cv_img: np.ndarray, w: int, h: int) -> QPixmap:
        h_img, w_img, ch = cv_img.shape
        rgb   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qimg  = QImage(rgb.data, w_img, h_img, ch * w_img, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg.copy()).scaled(
            w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ===============================================================
# MainWindow：主視窗  V4
# ===============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Hikvision Multi-Camera Stream  V4.0  ─  最多 20 台 POE 相機")
        self.resize(1920, 1080)
        self.showMaximized()          # 啟動即最大化

        self.device_list = MV_CC_DEVICE_INFO_LIST()

        # cam_id → {cam, thread}
        self._cameras: dict[int, dict] = {}

        # 相機顯示名稱清單
        self._cam_labels: list[str] = [f"Cam {i:02d}" for i in range(MAX_CAMERAS)]

        # slot_id → cam_id 對應表
        self._slot_to_cam: dict[int, int] = {i: i for i in range(MAX_CAMERAS)}

        self._setup_ui()
        self._apply_dark_theme()
        self.enum_devices()

        self.monitor_thread = MonitorThread()
        self.monitor_thread.stats_ready.connect(self._on_monitor_stats)
        self.monitor_thread.start()

    # ──────────────────────────────────────────────────────────
    # UI 建構
    # ──────────────────────────────────────────────────────────
    def _setup_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # ─────────────────────────────────────────────────────
        # 左側控制欄（固定寬 320px）
        # ─────────────────────────────────────────────────────
        left_widget = QWidget()
        left_widget.setFixedWidth(320)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)

        # 1. 硬體監控
        hw_group = QGroupBox("🖥  硬體監控")
        hw_layout = QVBoxLayout(hw_group)
        hw_layout.setSpacing(4)
        self.cpu_lbl     = QLabel("CPU:  -- %")
        self.ram_lbl     = QLabel("RAM:  -- %")
        self.gpu_lbl     = QLabel("GPU:  -- %")
        self.vram_lbl    = QLabel("VRAM: -- / -- MB")
        self.cam_cnt_lbl = QLabel("串流中：0 台")
        for lbl in (self.cpu_lbl, self.ram_lbl, self.gpu_lbl,
                    self.vram_lbl, self.cam_cnt_lbl):
            lbl.setStyleSheet("font-size:13px; padding:2px 0;")
            hw_layout.addWidget(lbl)
        left_layout.addWidget(hw_group)

        # 2. 相機控制
        ctrl_group = QGroupBox("📷  相機控制")
        ctrl_layout = QVBoxLayout(ctrl_group)
        ctrl_layout.setSpacing(6)

        self.enum_btn = QPushButton("🔍  刷新設備列表")
        self.enum_btn.setMinimumHeight(34)
        self.enum_btn.clicked.connect(self.enum_devices)
        ctrl_layout.addWidget(self.enum_btn)

        mode_lbl = QLabel("運算模式：")
        mode_lbl.setStyleSheet("font-size:13px;")
        ctrl_layout.addWidget(mode_lbl)
        mode_h = QHBoxLayout()
        self.cpu_radio  = QRadioButton("CPU")
        self.cuda_radio = QRadioButton("CUDA (GPU)")
        self.cpu_radio.setStyleSheet("font-size:13px;")
        self.cuda_radio.setStyleSheet("font-size:13px;")
        self.cpu_radio.setChecked(True)
        if not HAS_CUDA:
            self.cuda_radio.setEnabled(False)
            self.cuda_radio.setToolTip("未偵測到 CUDA")
        mode_grp = QButtonGroup(self)
        mode_grp.addButton(self.cpu_radio)
        mode_grp.addButton(self.cuda_radio)
        mode_h.addWidget(self.cpu_radio)
        mode_h.addWidget(self.cuda_radio)
        ctrl_layout.addLayout(mode_h)

        self.start_all_btn = QPushButton("▶  開啟全部相機")
        self.start_all_btn.setMinimumHeight(36)
        self.start_all_btn.clicked.connect(self.start_all)
        self.stop_all_btn = QPushButton("■  停止全部相機")
        self.stop_all_btn.setMinimumHeight(36)
        self.stop_all_btn.clicked.connect(self.stop_all)
        ctrl_layout.addWidget(self.start_all_btn)
        ctrl_layout.addWidget(self.stop_all_btn)
        left_layout.addWidget(ctrl_group)

        # 3. 全域串流參數
        param_group = QGroupBox("⚙  全域串流參數")
        param_layout = QVBoxLayout(param_group)
        param_layout.setSpacing(6)

        fps_lbl = QLabel("FPS 限制（每台）：")
        fps_lbl.setStyleSheet("font-size:13px;")
        param_layout.addWidget(fps_lbl)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(15)
        self.fps_spin.setMinimumHeight(30)
        self.fps_spin.setStyleSheet("font-size:13px;")
        self.fps_spin.valueChanged.connect(self._apply_fps)
        param_layout.addWidget(self.fps_spin)

        thresh_lbl = QLabel("相減靈敏度 (Threshold)：")
        thresh_lbl.setStyleSheet("font-size:13px;")
        param_layout.addWidget(thresh_lbl)
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(5, 255)
        self.thresh_slider.setValue(30)
        self.thresh_lbl_val = QLabel("30")
        self.thresh_lbl_val.setFixedWidth(34)
        self.thresh_lbl_val.setStyleSheet("font-size:13px;")
        self.thresh_slider.valueChanged.connect(
            lambda v: (self.thresh_lbl_val.setText(str(v)), self._apply_threshold(v)))
        th_h = QHBoxLayout()
        th_h.addWidget(self.thresh_slider)
        th_h.addWidget(self.thresh_lbl_val)
        param_layout.addLayout(th_h)

        self.sub_all_btn = QPushButton("🔁  全部影像相減 ON/OFF")
        self.sub_all_btn.setCheckable(True)
        self.sub_all_btn.setMinimumHeight(34)
        self.sub_all_btn.clicked.connect(self._toggle_all_subtraction)
        param_layout.addWidget(self.sub_all_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#2a2a45;")
        param_layout.addWidget(sep)

        alert_hdr = QLabel("⚠  全域覆蓋率 Alert 閾值：")
        alert_hdr.setStyleSheet("font-size:13px;")
        param_layout.addWidget(alert_hdr)
        alert_h = QHBoxLayout()
        self.global_alert_spin = QDoubleSpinBox()
        self.global_alert_spin.setRange(0.0, 100.0)
        self.global_alert_spin.setValue(50.0)
        self.global_alert_spin.setSuffix("%")
        self.global_alert_spin.setDecimals(1)
        self.global_alert_spin.setMinimumHeight(30)
        self.global_alert_spin.setStyleSheet("font-size:13px;")
        self.global_alert_spin.setToolTip("套用到所有相機的預設 Alert 閾值")
        self.global_alert_spin.valueChanged.connect(self._apply_global_alert)
        apply_alert_btn = QPushButton("套用至全部")
        apply_alert_btn.setFixedWidth(90)
        apply_alert_btn.setMinimumHeight(30)
        apply_alert_btn.clicked.connect(
            lambda: self._apply_global_alert(self.global_alert_spin.value()))
        alert_h.addWidget(self.global_alert_spin)
        alert_h.addWidget(apply_alert_btn)
        param_layout.addLayout(alert_h)
        left_layout.addWidget(param_group)

        # 4. 偵測到的設備
        dev_group = QGroupBox("📋  偵測到的設備")
        dev_layout = QVBoxLayout(dev_group)
        self.dev_list = QListWidget()
        self.dev_list.setStyleSheet("font-size:13px;")
        dev_layout.addWidget(self.dev_list)
        left_layout.addWidget(dev_group)

        left_layout.addStretch()
        splitter.addWidget(left_widget)

        # ─────────────────────────────────────────────────────
        # 右側：分頁式相機顯示（10 頁 × 2 台/頁，左右並排）
        # ─────────────────────────────────────────────────────
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(0)

        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #222240;
                background: #0c0c1c;
                border-radius: 5px;
            }
            QTabBar::tab {
                background: #181830;
                color: #7788bb;
                border: 1px solid #222240;
                padding: 8px 14px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-size: 13px;
                min-width: 80px;
            }
            QTabBar::tab:selected {
                background: #1c1c3c;
                color: #aaccff;
                border-bottom: 2px solid #4a72d9;
                font-weight: bold;
            }
            QTabBar::tab:hover { background: #202040; }
        """)

        self._cam_widgets: list[CameraWidget] = []

        for page in range(PAGE_COUNT):
            page_widget = QWidget()
            page_layout = QHBoxLayout(page_widget)   # ← 左右並排
            page_layout.setContentsMargins(6, 6, 6, 6)
            page_layout.setSpacing(8)

            for j in range(CAMS_PER_PAGE):
                slot_id = page * CAMS_PER_PAGE + j
                cw = CameraWidget(slot_id, self._cam_labels)
                cw.subtraction_toggled.connect(self._on_subtraction_toggled)
                cw.slot_cam_changed.connect(self._on_slot_cam_changed)
                page_layout.addWidget(cw, 1)      # stretch=1：各自佔半邊
                self._cam_widgets.append(cw)

            first = page * CAMS_PER_PAGE + 1
            last  = (page + 1) * CAMS_PER_PAGE
            self.tab_widget.addTab(
                page_widget, f"📷 P{page+1}  #{first:02d}-{last:02d}")

        right_layout.addWidget(self.tab_widget)
        splitter.addWidget(right_container)
        splitter.setSizes([320, 1600])

    # ──────────────────────────────────────────────────────────
    # 深色主題
    # ──────────────────────────────────────────────────────────
    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #0e0e1e;
                color: #ccd4e6;
                font-family: 'Noto Sans CJK TC', 'Segoe UI', 'Microsoft JhengHei', sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #252545;
                border-radius: 7px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: bold;
                font-size: 13px;
                color: #7799cc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
            }
            QPushButton {
                background: #1c1c35;
                border: 1px solid #383858;
                border-radius: 6px;
                padding: 6px 12px;
                color: #ccd4e6;
                font-size: 13px;
            }
            QPushButton:hover   { background: #272748; border-color: #5858aa; }
            QPushButton:pressed { background: #101025; }
            QPushButton:checked { background: #184028; border-color: #2ecc71; color: #80ffaa; }
            QPushButton:disabled{ background: #141428; color: #444466; }
            QSlider::groove:horizontal {
                height: 5px; background: #252545; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #5078d9; border-radius: 7px;
                width: 14px; height: 14px; margin: -5px 0;
            }
            QSpinBox, QDoubleSpinBox {
                background: #1c1c35; border: 1px solid #383858;
                border-radius: 5px; padding: 4px; font-size: 13px;
            }
            QScrollArea { border: none; }
            QListWidget {
                background: #16162e; border: 1px solid #252545;
                border-radius: 5px; font-size: 13px;
            }
            QListWidget::item { padding: 4px 6px; }
            QListWidget::item:selected { background: #252560; }
            QSplitter::handle { background: #222245; width: 5px; }
            QRadioButton { font-size: 13px; spacing: 6px; }
        """)

    # ──────────────────────────────────────────────────────────
    # 設備列舉
    # ──────────────────────────────────────────────────────────
    def enum_devices(self):
        self.dev_list.clear()
        self.device_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, self.device_list)
        if ret != 0:
            QMessageBox.warning(self, "錯誤", f"列舉設備失敗，ret=0x{ret:08X}")
            return

        n = self.device_list.nDeviceNum
        self._cam_labels = []

        for i in range(n):
            mvcc_dev_info = cast(
                self.device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents

            model, serial, ip = "", "", ""

            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                info   = mvcc_dev_info.SpecialInfo.stGigEInfo
                model  = "".join([chr(c) for c in info.chModelName   if c != 0]).strip()
                serial = "".join([chr(c) for c in info.chSerialNumber if c != 0]).strip()
                raw_ip = info.nCurrentIp
                ip     = (f"{(raw_ip>>24)&0xff}.{(raw_ip>>16)&0xff}"
                          f".{(raw_ip>>8)&0xff}.{raw_ip&0xff}")
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                info   = mvcc_dev_info.SpecialInfo.stUsb3VInfo
                model  = "".join([chr(c) for c in info.chModelName   if c != 0]).strip()
                serial = "".join([chr(c) for c in info.chSerialNumber if c != 0]).strip()

            if ip:
                label = f"{model}  [{ip}]"
            elif serial:
                label = f"{model}  S/N:{serial}"
            else:
                label = model or f"Cam{i:02d}"

            self._cam_labels.append(label)
            self.dev_list.addItem(f"[{i:02d}]  {label}")

        if n == 0:
            self.dev_list.addItem("（未偵測到設備）")
            return

        while len(self._cam_labels) < MAX_CAMERAS:
            self._cam_labels.append(f"Cam {len(self._cam_labels):02d}")

        for cw in self._cam_widgets:
            cw.set_cam_labels(self._cam_labels)

    # ──────────────────────────────────────────────────────────
    # 槽位與相機對應
    # ──────────────────────────────────────────────────────────
    def _on_slot_cam_changed(self, slot_id: int, cam_id: int):
        self._slot_to_cam[slot_id] = cam_id

    def _get_slot_for_cam(self, cam_id: int) -> int:
        for slot, cid in self._slot_to_cam.items():
            if cid == cam_id:
                return slot
        return -1

    # ──────────────────────────────────────────────────────────
    # 開啟 / 停止相機
    # ──────────────────────────────────────────────────────────
    def _open_camera(self, idx: int) -> bool:
        if idx in self._cameras:
            return True

        stDevInfo = cast(
            self.device_list.pDeviceInfo[idx], POINTER(MV_CC_DEVICE_INFO)).contents
        cam = MvCamera()
        if cam.MV_CC_CreateHandle(stDevInfo) != 0:
            return False
        if cam.MV_CC_OpenDevice() != 0:
            cam.MV_CC_DestroyHandle()
            return False

        cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if cam.MV_CC_StartGrabbing() != 0:
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            return False

        use_cuda = self.cuda_radio.isChecked() and HAS_CUDA
        t = CameraThread(cam, cam_id=idx, use_cuda=use_cuda)
        t.target_fps     = self.fps_spin.value()
        t.diff_threshold = self.thresh_slider.value()
        t.frame_ready.connect(self._on_frame_ready)
        t.start()

        self._cameras[idx] = {"cam": cam, "thread": t}

        slot = self._get_slot_for_cam(idx)
        if 0 <= slot < len(self._cam_widgets):
            self._cam_widgets[slot].status_lbl.setStyleSheet(
                "color:#2ecc71; font-size:16px;")
        return True

    def _close_camera(self, idx: int):
        if idx not in self._cameras:
            return
        entry = self._cameras.pop(idx)
        entry["thread"].stop()
        entry["cam"].MV_CC_StopGrabbing()
        entry["cam"].MV_CC_CloseDevice()
        entry["cam"].MV_CC_DestroyHandle()

        slot = self._get_slot_for_cam(idx)
        if 0 <= slot < len(self._cam_widgets):
            self._cam_widgets[slot].set_disconnected()

    def start_all(self):
        if self.cuda_radio.isChecked() and not HAS_CUDA:
            QMessageBox.warning(self, "CUDA 錯誤", "未偵測到 CUDA，已自動切換至 CPU 模式")
            self.cpu_radio.setChecked(True)

        n = min(self.device_list.nDeviceNum, MAX_CAMERAS)
        for i in range(n):
            self._open_camera(i)
        self.cam_cnt_lbl.setText(f"串流中：{len(self._cameras)} 台")

    def stop_all(self):
        for idx in list(self._cameras.keys()):
            self._close_camera(idx)
        self.cam_cnt_lbl.setText("串流中：0 台")

    # ──────────────────────────────────────────────────────────
    # 影像接收
    # ──────────────────────────────────────────────────────────
    def _on_frame_ready(self, cam_id: int, bgr: np.ndarray, diff: np.ndarray, coverage: float):
        slot = self._get_slot_for_cam(cam_id)
        if 0 <= slot < len(self._cam_widgets):
            self._cam_widgets[slot].update_frames(bgr, diff, coverage)

    # ──────────────────────────────────────────────────────────
    # 相減控制
    # ──────────────────────────────────────────────────────────
    def _on_subtraction_toggled(self, cam_id: int, enabled: bool):
        if cam_id in self._cameras:
            self._cameras[cam_id]["thread"].set_subtraction(enabled)

    def _toggle_all_subtraction(self, checked: bool):
        self.sub_all_btn.setText(
            "🔁  全部相減 ON" if checked else "🔁  全部影像相減 ON/OFF")
        for cw in self._cam_widgets:
            cw.sub_btn.setChecked(checked)
            cw.sub_btn.setText("相減 ON" if checked else "影像相減")
            cw._sub_enabled = checked
            cw.diff_lbl.setVisible(checked)
            cam_id = cw.current_cam_id()
            if cam_id in self._cameras:
                self._cameras[cam_id]["thread"].set_subtraction(checked)

    # ──────────────────────────────────────────────────────────
    # 全域 Alert 閾值
    # ──────────────────────────────────────────────────────────
    def _apply_global_alert(self, val: float):
        for cw in self._cam_widgets:
            cw.set_alert_threshold(val)

    # ──────────────────────────────────────────────────────────
    # 全域參數同步
    # ──────────────────────────────────────────────────────────
    def _apply_fps(self, val: int):
        for entry in self._cameras.values():
            entry["thread"].target_fps = val

    def _apply_threshold(self, val: int):
        for entry in self._cameras.values():
            entry["thread"].diff_threshold = val

    # ──────────────────────────────────────────────────────────
    # 硬體監控（背景執行緒回傳）
    # ──────────────────────────────────────────────────────────
    def _on_monitor_stats(self, cpu: float, ram: float, gpu: str, vram: str):
        self.cpu_lbl.setText(f"CPU:  {cpu:.1f}%")
        self.ram_lbl.setText(f"RAM:  {ram:.1f}%")
        if self.monitor_thread.has_nvidia:
            self.gpu_lbl.setText(f"GPU:  {gpu}%")
            self.vram_lbl.setText(f"VRAM: {vram}")
        self.cam_cnt_lbl.setText(f"串流中：{len(self._cameras)} 台")

    # ──────────────────────────────────────────────────────────
    # 關閉事件
    # ──────────────────────────────────────────────────────────
    def closeEvent(self, event):
        self.monitor_thread.stop()
        self.stop_all()
        event.accept()


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())