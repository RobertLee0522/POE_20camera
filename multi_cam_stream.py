"""
multi_cam_stream.py
====================
基於 V2.0.0 改良版 V3.0.0：
  - 分頁顯示：每頁 5 台相機（更大畫面）
  - 相機名稱顯示為「型號 + IP」或「產品序號」
  - 每個視窗槽位可透過下拉選單選擇要顯示哪台相機
  - 覆蓋率 Contour 計算，顯示面積比例，可設定 Alert 閾值
  - 全域硬體監控 (CPU / RAM / GPU / VRAM)
  - CPU / CUDA 雙模式運算引擎

作者：Robert Lee  版本：V3.0.0
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
    QTabWidget, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

import cv2

MAX_CAMERAS   = 20
CAMS_PER_PAGE = 5   # 每個分頁顯示 5 台相機
PAGE_COUNT    = MAX_CAMERAS // CAMS_PER_PAGE  # 4 頁

# -------------------------------------------------------
# 工具函式：判斷 CUDA 是否可用
# -------------------------------------------------------
def is_cuda_supported():
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
    stats_ready = pyqtSignal(float, float, str, str)  # cpu, ram, gpu_util, vram

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.has_nvidia = True

    def run(self):
        while self.running:
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            gpu_util, vram = "0", "0 / 0 MB"

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
# CameraThread：單一相機的取像執行緒
#   - subtraction_enabled : 即時開關影像相減，不需重啟執行緒
#   - use_cuda            : 選擇運算後端
# ===============================================================
class CameraThread(QThread):
    # 信號：傳出 (相機id, 原始BGR, 相減結果BGR, 覆蓋率%) 四個參數
    frame_ready = pyqtSignal(int, np.ndarray, np.ndarray, float)

    def __init__(self, cam_obj, cam_id: int, use_cuda: bool = False, parent=None):
        super().__init__(parent)
        self.cam    = cam_obj
        self.cam_id = cam_id
        self.use_cuda = use_cuda

        self.running = True
        self.target_fps    = 15
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

        # 預先配置 RGB 轉換的大容量記憶體（原 Payload 的 3 倍，足夠轉存 BGR8）
        max_rgb_size = nPayloadSize * 3
        pDataForRGB = (c_ubyte * max_rgb_size)()

        if self.use_cuda:
            gpu_frame    = cv2.cuda_GpuMat()
            gpu_bg_f32   = cv2.cuda_GpuMat()
            gpu_gray_f32 = cv2.cuda_GpuMat()
            gpu_bg_8u    = cv2.cuda_GpuMat()
            gaussian_filter = cv2.cuda.createGaussianFilter(
                cv2.CV_8UC1, cv2.CV_8UC1, (21, 21), 0)
            is_bg_initialized = False

        while self.running:
            t0 = time.time()

            ret = self.cam.MV_CC_GetOneFrameTimeout(pData, nPayloadSize, stFrameInfo)
            if ret == 0:
                stConvert = MV_CC_PIXEL_CONVERT_PARAM()
                stConvert.nWidth      = stFrameInfo.nWidth
                stConvert.nHeight     = stFrameInfo.nHeight
                stConvert.pSrcData    = cast(pData, POINTER(c_ubyte))
                stConvert.nSrcDataLen = stFrameInfo.nFrameLen
                stConvert.enSrcPixelType = stFrameInfo.enPixelType

                nRGBSize = stFrameInfo.nWidth * stFrameInfo.nHeight * 3
                
                # 如果突發需要更大記憶體，則重新配置（正常情況不會發生）
                if nRGBSize > max_rgb_size:
                    max_rgb_size = nRGBSize
                    pDataForRGB = (c_ubyte * max_rgb_size)()

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

                    # ── Contour 覆蓋率計算 ──────────────────────────
                    # 形態學處理：去噪 + 填補小空洞
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    cleaned = cv2.morphologyEx(thresh_res, cv2.MORPH_CLOSE, kernel)
                    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

                    contours, _ = cv2.findContours(
                        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    total_pixels = thresh_res.shape[0] * thresh_res.shape[1]
                    covered_pixels = sum(cv2.contourArea(c) for c in contours)
                    coverage_pct = (covered_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

                    # 在差異圖上繪製 contours（綠色外框）
                    diff_display = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
                    cv2.drawContours(diff_display, contours, -1, (0, 255, 100), 1)

                    # 顯示覆蓋率文字
                    cv2.putText(diff_display, f"Coverage: {coverage_pct:.1f}%",
                                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1)
                else:
                    diff_display = np.zeros_like(frame_bgr)
                    cv2.putText(diff_display, "Subtraction OFF",
                                (10, frame_bgr.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)

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
# CameraWidget：單一相機的顯示槽位（大畫面版）
#   - 上方：下拉選單選擇要顯示哪台相機
#   - 左側：原始影像；右側：相減+Contour 影像
#   - 底部：覆蓋率進度條 + 狀態
# ===============================================================
class CameraWidget(QFrame):
    subtraction_toggled = pyqtSignal(int, bool)   # (cam_id, enabled)
    slot_cam_changed    = pyqtSignal(int, int)     # (slot_id, cam_id)

    def __init__(self, slot_id: int, cam_labels: list, parent=None):
        """
        slot_id    : 此槽位編號 (0~19)
        cam_labels : 所有相機的顯示名稱清單（型號+IP 或序號）
        """
        super().__init__(parent)
        self.slot_id   = slot_id
        self.cam_labels = cam_labels
        self._current_cam_id = slot_id  # 預設：第 N 槽顯示第 N 台相機
        self._sub_enabled = False
        self._coverage_alert_threshold = 50.0  # 預設 50% 觸發 Alert
        self._alert_active = False
        self._setup_ui()

    def _setup_ui(self):
        self.setFrameShape(QFrame.Box)
        self.setLineWidth(1)
        self.setStyleSheet("""
            CameraWidget {
                border: 1px solid #3a3a50;
                border-radius: 5px;
                background: #13132a;
            }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 2, 4, 2)
        root.setSpacing(2)

        # ─ 單行工具列（緊湊）──────────────────────────────────
        bar = QHBoxLayout()
        bar.setSpacing(4)
        bar.setContentsMargins(0, 0, 0, 0)

        slot_lbl = QLabel(f"#{self.slot_id + 1:02d}")
        slot_lbl.setStyleSheet(
            "color:#5577bb; font-weight:bold; font-size:9px; min-width:22px;")
        bar.addWidget(slot_lbl)

        self.cam_combo = QComboBox()
        self.cam_combo.setStyleSheet("""
            QComboBox {
                background:#1a1a30; border:1px solid #3a3a5a;
                border-radius:3px; padding:0px 3px;
                font-size:9px; color:#a8c0ff;
                min-width:120px;
            }
            QComboBox::drop-down { border:none; width:12px; }
            QComboBox QAbstractItemView {
                background:#1a1a2e; selection-background-color:#2a2a5a;
                font-size:9px;
            }
        """)
        self.cam_combo.setFixedHeight(18)
        self.cam_combo.addItem("─ 未指派 ─", -1)
        for i, label in enumerate(self.cam_labels):
            self.cam_combo.addItem(label, i)
        if self.slot_id < len(self.cam_labels):
            self.cam_combo.setCurrentIndex(self.slot_id + 1)
        self.cam_combo.currentIndexChanged.connect(self._on_combo_changed)
        bar.addWidget(self.cam_combo, 1)

        self.cov_lbl = QLabel("0.0%")
        self.cov_lbl.setStyleSheet(
            "color:#55aadd; font-size:9px; min-width:34px; text-align:right;")
        bar.addWidget(self.cov_lbl)

        alert_icon = QLabel("⚠")
        alert_icon.setStyleSheet("color:#aa8822; font-size:9px;")
        bar.addWidget(alert_icon)
        self.alert_spin = QDoubleSpinBox()
        self.alert_spin.setRange(0.0, 100.0)
        self.alert_spin.setValue(50.0)
        self.alert_spin.setSuffix("%")
        self.alert_spin.setDecimals(0)
        self.alert_spin.setFixedWidth(58)
        self.alert_spin.setFixedHeight(18)
        self.alert_spin.setToolTip("覆蓋率超過此值觸發 Alert")
        self.alert_spin.setStyleSheet("""
            QDoubleSpinBox {
                background:#1a1a30; border:1px solid #3a3a5a;
                border-radius:3px; padding:0px 2px; font-size:9px;
            }
        """)
        self.alert_spin.valueChanged.connect(self._on_alert_threshold_changed)
        bar.addWidget(self.alert_spin)

        self.alert_lbl = QLabel()
        self.alert_lbl.setStyleSheet(
            "color:#ff4444; font-size:9px; font-weight:bold; min-width:48px;")
        bar.addWidget(self.alert_lbl)

        self.sub_btn = QPushButton("相減")
        self.sub_btn.setCheckable(True)
        self.sub_btn.setFixedSize(40, 18)
        self.sub_btn.setStyleSheet("""
            QPushButton {
                background:#222238; color:#666; border-radius:3px;
                font-size:8px; border:1px solid #3a3a55;
            }
            QPushButton:checked {
                background:#1a5a32; color:#7fff9a; border:1px solid #2ecc71;
            }
        """)
        self.sub_btn.clicked.connect(self._on_sub_toggled)
        bar.addWidget(self.sub_btn)

        self.status_lbl = QLabel("●")
        self.status_lbl.setStyleSheet("color:#444; font-size:10px;")
        self.status_lbl.setToolTip("未連線")
        bar.addWidget(self.status_lbl)

        root.addLayout(bar)

        # ─ 影像區：主畫面（全寬）+ 右上角 PiP 差異縮圖 ──────
        img_container = QWidget()
        img_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        img_container.setMinimumHeight(60)
        img_container.setStyleSheet("background:#080814; border-radius:3px;")

        self.orig_lbl = QLabel(img_container)
        self.orig_lbl.setAlignment(Qt.AlignCenter)
        self.orig_lbl.setStyleSheet("background:transparent; color:#333; font-size:9px;")
        self.orig_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.orig_lbl.setText("等待連線...")

        self.diff_lbl = QLabel(img_container)
        self.diff_lbl.setAlignment(Qt.AlignCenter)
        self.diff_lbl.setStyleSheet("""
            background:#0d1220;
            border:1px solid #2a3a5a;
            border-radius:2px;
            color:#444; font-size:7px;
        """)
        self.diff_lbl.setText("差異")
        self.diff_lbl.setVisible(False)

        root.addWidget(img_container, 1)

    # ─ 槽位對應的相機 ID ────────────────────────────────────────
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

    # ─ img_container 內部子 widget 位置隨視窗縮放 ──────────────
    def resizeEvent(self, event):
        super().resizeEvent(event)
        container = self.orig_lbl.parent()
        if container is None:
            return
        cw = container.width()
        ch = container.height()
        # 主畫面佔滿整個 container
        self.orig_lbl.setGeometry(0, 0, cw, ch)
        # PiP：右上角，寬度 = 容器 25%，高度按 16:9 估算，最小 60×34
        pip_w = max(int(cw * 0.25), 80)
        pip_h = max(int(pip_w * 9 / 16), 46)
        margin = 4
        self.diff_lbl.setGeometry(cw - pip_w - margin, margin, pip_w, pip_h)

    # ─ 相減按鈕 ─────────────────────────────────────────────────
    def _on_sub_toggled(self, checked: bool):
        self._sub_enabled = checked
        self.sub_btn.setText("相減 ON" if checked else "相減")
        self.diff_lbl.setVisible(checked)
        self.subtraction_toggled.emit(self._current_cam_id, checked)

    # ─ Alert 閾值 ───────────────────────────────────────────────
    def _on_alert_threshold_changed(self, val: float):
        self._coverage_alert_threshold = val

    def set_alert_threshold(self, val: float):
        self.alert_spin.blockSignals(True)
        self.alert_spin.setValue(val)
        self._coverage_alert_threshold = val
        self.alert_spin.blockSignals(False)

    # ─ 影像更新 ─────────────────────────────────────────────────
    def update_frames(self, bgr: np.ndarray, diff: np.ndarray, coverage: float):
        # 主畫面：填滿 orig_lbl
        w = self.orig_lbl.width()
        h = self.orig_lbl.height()
        if w > 0 and h > 0:
            self.orig_lbl.setPixmap(self._to_pixmap(bgr, w, h))

        # PiP 差異畫面（只有相減開啟時可見）
        if self._sub_enabled:
            pw = self.diff_lbl.width()
            ph = self.diff_lbl.height()
            if pw > 0 and ph > 0:
                self.diff_lbl.setPixmap(self._to_pixmap(diff, pw, ph))

        # 覆蓋率
        self.cov_lbl.setText(f"{coverage:.1f}%")

        # Alert
        if coverage >= self._coverage_alert_threshold:
            self._alert_active = True
            self.alert_lbl.setText(f"⚠ >{self._coverage_alert_threshold:.0f}%")
            self.setStyleSheet("""
                CameraWidget {
                    border: 2px solid #ff4444;
                    border-radius: 5px;
                    background: #1e0808;
                }
            """)
        else:
            self._alert_active = False
            self.alert_lbl.setText("")
            self.setStyleSheet("""
                CameraWidget {
                    border: 1px solid #3a3a50;
                    border-radius: 5px;
                    background: #13132a;
                }
            """)

        self.status_lbl.setText("●")
        self.status_lbl.setStyleSheet("color:#2ecc71; font-size:10px;")
        self.status_lbl.setToolTip("串流中")

    def set_disconnected(self):
        self.orig_lbl.setText("已停止")
        self.orig_lbl.setPixmap(QPixmap())
        self.diff_lbl.setVisible(False)
        self.status_lbl.setText("●")
        self.status_lbl.setStyleSheet("color:#e74c3c; font-size:10px;")
        self.status_lbl.setToolTip("已停止")
        self.cov_lbl.setText("--")
        self.alert_lbl.setText("")
        self.setStyleSheet("""
            CameraWidget {
                border: 1px solid #3a3a50;
                border-radius: 5px;
                background: #13132a;
            }
        """)

    @staticmethod
    def _to_pixmap(cv_img: np.ndarray, w: int, h: int) -> QPixmap:
        h_img, w_img, ch = cv_img.shape
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w_img, h_img, ch * w_img, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg.copy()).scaled(
            w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ===============================================================
# MainWindow：主視窗
# ===============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hikvision Multi-Camera Stream  V3.0  ─  最多 20 台 POE 相機")
        self.resize(1720, 1000)

        self.device_list = MV_CC_DEVICE_INFO_LIST()
        self.has_nvidia  = True

        # cam_id → {cam, thread}
        self._cameras: dict[int, dict] = {}

        # 相機顯示名稱清單（列舉後填入）
        self._cam_labels: list[str] = [f"Cam {i:02d}" for i in range(MAX_CAMERAS)]

        # slot_id → cam_id 對應表（由各 CameraWidget 的 combo 決定）
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

        # ── 左側控制欄 ────────────────────────────────────────
        left_widget = QWidget()
        left_widget.setFixedWidth(290)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        # 1. 硬體監控
        hw_group = QGroupBox("🖥  硬體監控")
        hw_layout = QVBoxLayout(hw_group)
        self.cpu_lbl  = QLabel("CPU:  0%")
        self.ram_lbl  = QLabel("RAM:  0%")
        self.gpu_lbl  = QLabel("GPU:  0%")
        self.vram_lbl = QLabel("VRAM: 0 / 0 MB")
        self.cam_cnt_lbl = QLabel("串流中：0 台")
        for lbl in (self.cpu_lbl, self.ram_lbl, self.gpu_lbl,
                    self.vram_lbl, self.cam_cnt_lbl):
            hw_layout.addWidget(lbl)
        left_layout.addWidget(hw_group)

        # 2. 相機控制
        ctrl_group = QGroupBox("📷  相機控制")
        ctrl_layout = QVBoxLayout(ctrl_group)

        self.enum_btn = QPushButton("🔍  刷新設備列表")
        self.enum_btn.clicked.connect(self.enum_devices)
        ctrl_layout.addWidget(self.enum_btn)

        ctrl_layout.addWidget(QLabel("運算模式："))
        mode_h = QHBoxLayout()
        self.cpu_radio  = QRadioButton("CPU")
        self.cuda_radio = QRadioButton("CUDA (GPU)")
        self.cpu_radio.setChecked(True)
        mode_grp = QButtonGroup(self)
        mode_grp.addButton(self.cpu_radio)
        mode_grp.addButton(self.cuda_radio)
        mode_h.addWidget(self.cpu_radio)
        mode_h.addWidget(self.cuda_radio)
        ctrl_layout.addLayout(mode_h)

        self.start_all_btn = QPushButton("▶  開啟全部相機")
        self.start_all_btn.clicked.connect(self.start_all)
        self.stop_all_btn  = QPushButton("■  停止全部相機")
        self.stop_all_btn.clicked.connect(self.stop_all)
        ctrl_layout.addWidget(self.start_all_btn)
        ctrl_layout.addWidget(self.stop_all_btn)
        left_layout.addWidget(ctrl_group)

        # 3. 全域串流參數
        param_group = QGroupBox("⚙  全域串流參數")
        param_layout = QVBoxLayout(param_group)

        param_layout.addWidget(QLabel("FPS 限制（每台）："))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(15)
        self.fps_spin.valueChanged.connect(self._apply_fps)
        param_layout.addWidget(self.fps_spin)

        param_layout.addWidget(QLabel("相減靈敏度 (Threshold)："))
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(5, 255)
        self.thresh_slider.setValue(30)
        self.thresh_lbl = QLabel("30")
        self.thresh_slider.valueChanged.connect(
            lambda v: (self.thresh_lbl.setText(str(v)), self._apply_threshold(v)))
        th_h = QHBoxLayout()
        th_h.addWidget(self.thresh_slider)
        th_h.addWidget(self.thresh_lbl)
        param_layout.addLayout(th_h)

        self.sub_all_btn = QPushButton("🔁  全部相減 ON/OFF")
        self.sub_all_btn.setCheckable(True)
        self.sub_all_btn.clicked.connect(self._toggle_all_subtraction)
        param_layout.addWidget(self.sub_all_btn)

        # 全域 Alert 閾值
        alert_sep = QFrame()
        alert_sep.setFrameShape(QFrame.HLine)
        alert_sep.setStyleSheet("color:#3a3a5a;")
        param_layout.addWidget(alert_sep)
        param_layout.addWidget(QLabel("⚠  全域覆蓋率 Alert 閾值："))
        alert_h = QHBoxLayout()
        self.global_alert_spin = QDoubleSpinBox()
        self.global_alert_spin.setRange(0.0, 100.0)
        self.global_alert_spin.setValue(50.0)
        self.global_alert_spin.setSuffix("%")
        self.global_alert_spin.setDecimals(1)
        self.global_alert_spin.setToolTip("套用到所有相機的預設 Alert 閾值")
        self.global_alert_spin.valueChanged.connect(self._apply_global_alert)
        apply_alert_btn = QPushButton("套用至全部")
        apply_alert_btn.setFixedWidth(80)
        apply_alert_btn.clicked.connect(
            lambda: self._apply_global_alert(self.global_alert_spin.value()))
        alert_h.addWidget(self.global_alert_spin)
        alert_h.addWidget(apply_alert_btn)
        param_layout.addLayout(alert_h)

        left_layout.addWidget(param_group)

        # 4. 設備清單
        dev_group = QGroupBox("📋  偵測到的設備")
        dev_layout = QVBoxLayout(dev_group)
        self.dev_list = QListWidget()
        self.dev_list.setMaximumHeight(160)
        dev_layout.addWidget(self.dev_list)
        left_layout.addWidget(dev_group)

        left_layout.addStretch()
        splitter.addWidget(left_widget)

        # ── 右側：分頁式相機顯示 ──────────────────────────────
        right_container = QWidget()
        right_layout    = QVBoxLayout(right_container)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #2a2a45;
                background: #0f0f1e;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #1a1a2e;
                color: #8899cc;
                border: 1px solid #2a2a45;
                padding: 6px 18px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                min-width: 100px;
            }
            QTabBar::tab:selected {
                background: #1e1e3e;
                color: #a0c4ff;
                border-bottom: 2px solid #5a82d9;
            }
            QTabBar::tab:hover { background: #22224a; }
        """)

        # 預先建立 MAX_CAMERAS 個 CameraWidget，分配到各分頁
        self._cam_widgets: list[CameraWidget] = []

        for page in range(PAGE_COUNT):
            page_widget = QWidget()
            page_layout = QVBoxLayout(page_widget)  # 5台垂直堆疊，各佔滿寬度
            page_layout.setContentsMargins(4, 4, 4, 4)
            page_layout.setSpacing(4)

            for j in range(CAMS_PER_PAGE):
                slot_id = page * CAMS_PER_PAGE + j
                cw = CameraWidget(slot_id, self._cam_labels)
                cw.subtraction_toggled.connect(self._on_subtraction_toggled)
                cw.slot_cam_changed.connect(self._on_slot_cam_changed)
                page_layout.addWidget(cw)
                self._cam_widgets.append(cw)

            cam_range = f"Cam {page * CAMS_PER_PAGE + 1:02d} – {(page + 1) * CAMS_PER_PAGE:02d}"
            self.tab_widget.addTab(page_widget, f"📷 頁{page + 1}  {cam_range}")

        right_layout.addWidget(self.tab_widget)
        splitter.addWidget(right_container)
        splitter.setSizes([290, 1430])

    # ──────────────────────────────────────────────────────────
    # 深色主題
    # ──────────────────────────────────────────────────────────
    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #0f0f1e;
                color: #c8d0e0;
                font-family: 'Segoe UI', sans-serif;
                font-size: 11px;
            }
            QGroupBox {
                border: 1px solid #2a2a45;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 8px;
                font-weight: bold;
                color: #8899cc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
            }
            QPushButton {
                background: #1e1e35;
                border: 1px solid #3a3a5a;
                border-radius: 5px;
                padding: 5px 10px;
                color: #c8d0e0;
            }
            QPushButton:hover  { background: #292945; border-color: #5a5aaa; }
            QPushButton:pressed{ background: #12122a; }
            QPushButton:checked{ background: #1a4a2a; border-color: #2ecc71; color: #7fff9a; }
            QSlider::groove:horizontal {
                height: 4px; background: #2a2a45; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #5a82d9; border-radius: 6px;
                width: 12px; height: 12px; margin: -4px 0;
            }
            QSpinBox, QDoubleSpinBox {
                background: #1e1e35; border: 1px solid #3a3a5a;
                border-radius: 4px; padding: 3px;
            }
            QScrollArea { border: none; }
            QListWidget {
                background: #1a1a2e; border: 1px solid #2a2a45; border-radius:4px;
            }
            QListWidget::item:selected { background: #2a2a5a; }
            QSplitter::handle { background: #2a2a45; width: 4px; }
        """)

    # ──────────────────────────────────────────────────────────
    # 設備列舉（更新名稱為型號+IP）
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

            model = ""
            serial = ""
            ip = ""

            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                info = mvcc_dev_info.SpecialInfo.stGigEInfo
                model = "".join([chr(c) for c in info.chModelName if c != 0]).strip()
                serial = "".join([chr(c) for c in info.chSerialNumber if c != 0]).strip()
                raw_ip = info.nCurrentIp
                ip = f"{(raw_ip>>24)&0xff}.{(raw_ip>>16)&0xff}.{(raw_ip>>8)&0xff}.{raw_ip&0xff}"
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                info = mvcc_dev_info.SpecialInfo.stUsb3VInfo
                model = "".join([chr(c) for c in info.chModelName if c != 0]).strip()
                serial = "".join([chr(c) for c in info.chSerialNumber if c != 0]).strip()
            
            # 建立顯示標籤：優先顯示型號+IP，若有序號則附上

            if ip:
                label = f"{model}  [{ip}]"
            elif serial:
                label = f"{model}  S/N:{serial}"
            else:
                label = f"{model or f'Cam{i:02d}'}"

            self._cam_labels.append(label)
            self.dev_list.addItem(f"[{i:02d}] {label}")

        if n == 0:
            self.dev_list.addItem("（未偵測到設備）")
            return

        # 補足未對應的相機名稱（若不足 MAX_CAMERAS）
        while len(self._cam_labels) < MAX_CAMERAS:
            self._cam_labels.append(f"Cam {len(self._cam_labels):02d}")

        # 更新所有 CameraWidget 的下拉選單
        for cw in self._cam_widgets:
            cw.set_cam_labels(self._cam_labels)

    # ──────────────────────────────────────────────────────────
    # 槽位與相機對應關係
    # ──────────────────────────────────────────────────────────
    def _on_slot_cam_changed(self, slot_id: int, cam_id: int):
        """使用者更改某槽位對應的相機時觸發。"""
        self._slot_to_cam[slot_id] = cam_id

    def _get_slot_for_cam(self, cam_id: int) -> int:
        """回傳某 cam_id 目前被分配到的槽位 (-1 表示未顯示)。"""
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

        # 更新對應槽位的狀態標籤
        slot = self._get_slot_for_cam(idx)
        if slot >= 0 and slot < len(self._cam_widgets):
            self._cam_widgets[slot].status_lbl.setText("● 串流中")
            self._cam_widgets[slot].status_lbl.setStyleSheet(
                "color:#2ecc71; font-size:9px;")
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
        if slot >= 0 and slot < len(self._cam_widgets):
            self._cam_widgets[slot].set_disconnected()

    def start_all(self):
        use_cuda = self.cuda_radio.isChecked()
        if use_cuda and not HAS_CUDA:
            QMessageBox.warning(self, "CUDA 錯誤", "未偵測到 CUDA，自動切換至 CPU 模式")
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
    # 影像接收與顯示（根據 slot→cam 對應表分發）
    # ──────────────────────────────────────────────────────────
    def _on_frame_ready(self, cam_id: int, bgr: np.ndarray, diff: np.ndarray, coverage: float):
        slot = self._get_slot_for_cam(cam_id)
        if slot >= 0 and slot < len(self._cam_widgets):
            self._cam_widgets[slot].update_frames(bgr, diff, coverage)

    # ──────────────────────────────────────────────────────────
    # 影像相減開關
    # ──────────────────────────────────────────────────────────
    def _on_subtraction_toggled(self, cam_id: int, enabled: bool):
        if cam_id in self._cameras:
            self._cameras[cam_id]["thread"].set_subtraction(enabled)

    def _toggle_all_subtraction(self, checked: bool):
        self.sub_all_btn.setText(
            "🔁  全部相減 ON" if checked else "🔁  全部相減 OFF")
        for i, cw in enumerate(self._cam_widgets):
            cw.sub_btn.setChecked(checked)
            cw.sub_btn.setText("相減 ON" if checked else "相減")
            cam_id = cw.current_cam_id()
            if cam_id in self._cameras:
                self._cameras[cam_id]["thread"].set_subtraction(checked)

    # ──────────────────────────────────────────────────────────
    # 全域 Alert 閾值套用
    # ──────────────────────────────────────────────────────────
    def _apply_global_alert(self, val: float):
        for cw in self._cam_widgets:
            cw.set_alert_threshold(val)

    # ──────────────────────────────────────────────────────────
    # 全域參數即時同步
    # ──────────────────────────────────────────────────────────
    def _apply_fps(self, val: int):
        for entry in self._cameras.values():
            entry["thread"].target_fps = val

    def _apply_threshold(self, val: int):
        for entry in self._cameras.values():
            entry["thread"].diff_threshold = val

    # ──────────────────────────────────────────────────────────
    # 硬體監控 (背景執行緒回傳)
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