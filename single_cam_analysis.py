import sys
import ctypes
from ctypes import *
import numpy as np
import cv2
import time
import os
import platform
import psutil

# 支援 Windows / Linux 跨平台的 MVS SDK 路徑匯入
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
except ImportError:
    print("【嚴重錯誤】: 找不到海康威視 MVS Python SDK，請確認 MVS 已正確安裝。")
    sys.exit(1)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QGroupBox, QSpinBox, QMessageBox,
    QGridLayout, QDoubleSpinBox, QCheckBox, QFrame, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont


class CameraThread(QThread):
    # 發送處理後的影像與統計數據
    frame_ready = pyqtSignal(np.ndarray, dict)

    def __init__(self, cam_obj, parent=None):
        super().__init__(parent)
        self.cam = cam_obj
        self.running = True
        self.target_width = 0   # 0 = 不縮放，保持原始相機分辨率
        self.target_fps = 15

    def set_fps(self, fps: int):
        self.target_fps = fps

    def set_resize_width(self, width_px: int):
        """width_px=0 表示不縮放"""
        self.target_width = width_px

    def run(self):
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))

        max_rgb_size = 0
        pDataForRGB = None

        while self.running:
            t0 = time.time()

            # SDK 自行管理取像 buffer，不需要預先知道 PayloadSize
            ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if ret != 0:
                elapsed = time.time() - t0
                time.sleep(max(1.0 / self.target_fps - elapsed, 0.001))
                continue

            fi  = stOutFrame.stFrameInfo
            nW  = fi.nWidth
            nH  = fi.nHeight
            nRGBSize = nW * nH * 3
            frame_bgr = None

            # ── 先嘗試 SDK 轉換 BayerRG8 → BGR8 ──────────────
            if nRGBSize > max_rgb_size:
                max_rgb_size = nRGBSize
                pDataForRGB  = (c_ubyte * max_rgb_size)()

            stConvert = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvert), 0, sizeof(stConvert))
            stConvert.nWidth         = nW
            stConvert.nHeight        = nH
            stConvert.pSrcData       = stOutFrame.pBufAddr
            stConvert.nSrcDataLen    = fi.nFrameLen
            stConvert.enSrcPixelType = fi.enPixelType
            stConvert.enDstPixelType = PixelType_Gvsp_BGR8_Packed
            stConvert.pDstBuffer     = cast(pDataForRGB, POINTER(c_ubyte))
            stConvert.nDstBufferSize = max_rgb_size

            ret_conv = self.cam.MV_CC_ConvertPixelType(stConvert)

            if ret_conv == 0:
                # SDK 轉換成功
                arr = np.frombuffer(pDataForRGB, count=nRGBSize, dtype=np.uint8)
                frame_bgr = arr.reshape((nH, nW, 3)).copy()
            else:
                # Fallback：先把原始 Bayer 資料複製出來，再釋放 buffer
                raw_expected = nW * nH
                if fi.nFrameLen >= raw_expected:
                    raw_buf = (c_ubyte * raw_expected)()
                    ctypes.memmove(raw_buf, stOutFrame.pBufAddr, raw_expected)
                    raw_arr = np.frombuffer(raw_buf, dtype=np.uint8).reshape((nH, nW))
                    frame_bgr = cv2.cvtColor(raw_arr, cv2.COLOR_BayerRG2BGR)
                    print(f"[INFO] SDK轉換失敗(0x{ret_conv & 0xFFFFFFFF:08X})，使用OpenCV Bayer fallback "
                          f"FrameLen={fi.nFrameLen} W={nW} H={nH}")
                else:
                    print(f"[WARN] FrameLen={fi.nFrameLen} < 期望={raw_expected}，跳過此幀")

            # 釋放 SDK 內部 buffer（必須在 memmove/copy 後才釋放）
            self.cam.MV_CC_FreeImageBuffer(stOutFrame)

            if frame_bgr is None:
                elapsed = time.time() - t0
                time.sleep(max(1.0 / self.target_fps - elapsed, 0.001))
                continue

            # ── 共用後處理 ─────────────────────────────────────
            h, w = frame_bgr.shape[:2]
            if self.target_width > 0 and self.target_width != w:
                scale = self.target_width / w
                new_w = self.target_width
                new_h = max(1, int(h * scale))
                proc_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                proc_bgr = frame_bgr
                new_w, new_h = w, h

            # RGB 通道平均
            b_mean = float(np.mean(proc_bgr[:, :, 0]))
            g_mean = float(np.mean(proc_bgr[:, :, 1]))
            r_mean = float(np.mean(proc_bgr[:, :, 2]))

            # HLS（含光強 L）
            frame_hls = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2HLS)
            h_mean = float(np.mean(frame_hls[:, :, 0]))
            l_mean = float(np.mean(frame_hls[:, :, 1]))
            s_mean = float(np.mean(frame_hls[:, :, 2]))

            # 相機曝光值
            exposure_obj = MVCC_FLOATVALUE()
            self.cam.MV_CC_GetFloatValue("ExposureTime", exposure_obj)

            stats = {
                "R": r_mean, "G": g_mean, "B": b_mean,
                "H": h_mean, "S": s_mean, "L": l_mean,
                "Intensity": l_mean,
                "Exposure": exposure_obj.fCurValue,
                "Resolution": f"{new_w} x {new_h}"
            }

            self.frame_ready.emit(proc_bgr, stats)

            elapsed = time.time() - t0
            time.sleep(max(1.0 / self.target_fps - elapsed, 0.001))

    def stop(self):
        self.running = False
        self.wait()



class SingleCamAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Single Camera Analysis - 基礎影像分析")
        self.resize(1200, 800)
        self.setStyleSheet("background-color: #121212; color: #E0E0E0;")
        
        self.cam = None
        self.is_streaming = False
        self.stream_thread = None
        self._last_frame_time = 0.0
        self.device_list = MV_CC_DEVICE_INFO_LIST()

        self._setup_ui()
        
        self.timer_hw = QTimer(self)
        self.timer_hw.timeout.connect(self.update_hw_stats)
        self.timer_hw.start(1000)

        # 啟動後自動掃描並連線第一台相機
        QTimer.singleShot(200, self.auto_connect)

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ── 左側設定面板 ──
        left_panel = QWidget()
        left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout(left_panel)

        # 設備控制區
        dev_group = QGroupBox("相機連線")
        dev_group.setStyleSheet("QGroupBox { border: 1px solid #333; margin-top: 10px; }")
        dev_layout = QVBoxLayout(dev_group)
        
        scan_layout = QHBoxLayout()
        self.combo_cams = QComboBox()
        self.btn_scan = QPushButton("掃描相機")
        self.btn_scan.clicked.connect(self.scan_cameras)
        scan_layout.addWidget(self.combo_cams, stretch=1)
        scan_layout.addWidget(self.btn_scan)

        self.btn_open = QPushButton("連線選定相機")
        self.btn_open.clicked.connect(self.open_camera)
        self.btn_start = QPushButton("開始讀取")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.toggle_stream)
        
        dev_layout.addLayout(scan_layout)
        dev_layout.addWidget(self.btn_open)
        dev_layout.addWidget(self.btn_start)
        left_layout.addWidget(dev_group)

        # 分析數據顯示區
        self.stats_group = QGroupBox("即時影像統計資料")
        self.stats_group.setStyleSheet("QGroupBox { border: 1px solid #333; margin-top: 10px; }")
        stats_layout = QGridLayout(self.stats_group)
        
        self.lbl_r = QLabel("R 平均: --")
        self.lbl_g = QLabel("G 平均: --")
        self.lbl_b = QLabel("B 平均: --")
        
        self.lbl_h = QLabel("H (Hue): --")
        self.lbl_s = QLabel("S (Sat): --")
        self.lbl_l = QLabel("L (Light): --")
        
        self.lbl_intensity = QLabel("平均光強: --")
        self.lbl_intensity.setStyleSheet("color: #FFD700; font-weight: bold;")
        
        self.lbl_exposure = QLabel("曝光值(us): --")
        self.lbl_res = QLabel("當前解析度: --")
        
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_cpu = QLabel("CPU: --%")
        self.lbl_ram = QLabel("RAM: --%")

        stats_layout.addWidget(self.lbl_r, 0, 0)
        stats_layout.addWidget(self.lbl_h, 0, 1)
        stats_layout.addWidget(self.lbl_g, 1, 0)
        stats_layout.addWidget(self.lbl_s, 1, 1)
        stats_layout.addWidget(self.lbl_b, 2, 0)
        stats_layout.addWidget(self.lbl_l, 2, 1)
        
        stats_layout.addWidget(QFrame(), 3, 0, 1, 2) # Separator
        
        stats_layout.addWidget(self.lbl_intensity, 4, 0, 1, 2)
        stats_layout.addWidget(self.lbl_exposure, 5, 0, 1, 2)
        stats_layout.addWidget(self.lbl_res, 6, 0, 1, 2)
        
        stats_layout.addWidget(QFrame(), 7, 0, 1, 2) # Separator
        
        stats_layout.addWidget(self.lbl_fps, 8, 0, 1, 2)
        stats_layout.addWidget(self.lbl_cpu, 9, 0)
        stats_layout.addWidget(self.lbl_ram, 9, 1)
        
        for i in range(stats_layout.count()):
            widget = stats_layout.itemAt(i).widget()
            if isinstance(widget, QLabel):
                widget.setStyleSheet("font-size: 14px; padding: 4px;")

        left_layout.addWidget(self.stats_group)

        # 影像調整控制區
        ctrl_group = QGroupBox("影像控制")
        ctrl_group.setStyleSheet("QGroupBox { border: 1px solid #333; margin-top: 10px; }")
        ctrl_layout = QVBoxLayout(ctrl_group)

        # 曝光控制 (手動)
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("曝光 (us):"))
        self.spin_exposure = QSpinBox()
        self.spin_exposure.setRange(20, 100000)
        self.spin_exposure.setSingleStep(1000)
        self.spin_exposure.setValue(10000)
        self.btn_set_exp = QPushButton("設置")
        self.btn_set_exp.clicked.connect(self.set_exposure)
        exp_layout.addWidget(self.spin_exposure)
        exp_layout.addWidget(self.btn_set_exp)
        ctrl_layout.addLayout(exp_layout)

        # FPS 控制 (硬體/軟體)
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 100)
        self.spin_fps.setSingleStep(1)
        self.spin_fps.setValue(15)
        self.btn_set_fps = QPushButton("設置")
        self.btn_set_fps.clicked.connect(self.set_fps)
        fps_layout.addWidget(self.spin_fps)
        fps_layout.addWidget(self.btn_set_fps)
        ctrl_layout.addLayout(fps_layout)

        # 白平衡控制
        wb_layout = QHBoxLayout()
        self.btn_auto_wb = QPushButton("執行單次自動白平衡")
        self.btn_auto_wb.clicked.connect(self.auto_white_balance)
        wb_layout.addWidget(self.btn_auto_wb)
        ctrl_layout.addLayout(wb_layout)

        # ─── ROI 設定（硬體感測區域）───
        roi_group = QGroupBox("ROI 設定")
        roi_group.setStyleSheet("QGroupBox { border: 1px solid #444; margin-top: 10px; }")
        roi_grid = QGridLayout(roi_group)
        roi_grid.setSpacing(6)

        roi_grid.addWidget(QLabel("Offset X:"), 0, 0)
        self.spin_roi_x = QSpinBox()
        self.spin_roi_x.setRange(0, 9999)
        self.spin_roi_x.setSingleStep(4)
        self.spin_roi_x.setValue(0)
        roi_grid.addWidget(self.spin_roi_x, 0, 1)

        roi_grid.addWidget(QLabel("Offset Y:"), 1, 0)
        self.spin_roi_y = QSpinBox()
        self.spin_roi_y.setRange(0, 9999)
        self.spin_roi_y.setSingleStep(4)
        self.spin_roi_y.setValue(0)
        roi_grid.addWidget(self.spin_roi_y, 1, 1)

        roi_grid.addWidget(QLabel("Width:"), 2, 0)
        self.spin_roi_w = QSpinBox()
        self.spin_roi_w.setRange(16, 9999)
        self.spin_roi_w.setSingleStep(4)
        self.spin_roi_w.setValue(2200)
        roi_grid.addWidget(self.spin_roi_w, 2, 1)

        roi_grid.addWidget(QLabel("Height:"), 3, 0)
        self.spin_roi_h = QSpinBox()
        self.spin_roi_h.setRange(16, 9999)
        self.spin_roi_h.setSingleStep(4)
        self.spin_roi_h.setValue(2048)
        roi_grid.addWidget(self.spin_roi_h, 3, 1)

        roi_btn_layout = QHBoxLayout()
        self.btn_apply_roi = QPushButton("套用 ROI")
        self.btn_apply_roi.clicked.connect(self.apply_roi)
        self.btn_reset_roi = QPushButton("重置全圖")
        self.btn_reset_roi.clicked.connect(self.reset_roi)
        roi_btn_layout.addWidget(self.btn_apply_roi)
        roi_btn_layout.addWidget(self.btn_reset_roi)
        roi_grid.addLayout(roi_btn_layout, 4, 0, 1, 2)

        ctrl_layout.addWidget(roi_group)

        left_layout.addWidget(ctrl_group)
        left_layout.addStretch()

        # ── 右側影像區 ──
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.image_label = QLabel("無影像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #000; border: 2px solid #555;")
        self.image_label.setMinimumSize(640, 480)
        right_layout.addWidget(self.image_label)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)

    def scan_cameras(self):
        self.combo_cams.clear()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, self.device_list)
        if ret != 0 or self.device_list.nDeviceNum == 0:
            self.combo_cams.addItem("未找到相機")
            return
            
        for i in range(self.device_list.nDeviceNum):
            device = cast(self.device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if device.nTLayerType == MV_GIGE_DEVICE:
                ip_bytes = device.SpecialInfo.stGigEInfo.nCurrentIp
                ip = f"{(ip_bytes & 0xff000000) >> 24}.{(ip_bytes & 0x00ff0000) >> 16}.{(ip_bytes & 0x0000ff00) >> 8}.{ip_bytes & 0x000000ff}"
                model = bytes(device.SpecialInfo.stGigEInfo.chModelName).replace(b'\x00', b'').decode('utf-8', 'ignore')
                self.combo_cams.addItem(f"[{i}] {model} ({ip})")
            else:
                model = bytes(device.SpecialInfo.stUsb3VInfo.chModelName).replace(b'\x00', b'').decode('utf-8', 'ignore')
                self.combo_cams.addItem(f"[{i}] USB: {model}")

    def open_camera(self):
        idx = self.combo_cams.currentIndex()
        if idx < 0 or self.device_list.nDeviceNum == 0:
            QMessageBox.warning(self, "錯誤", "請先掃描並選擇一台有效的相機！")
            return

        stDeviceList = cast(self.device_list.pDeviceInfo[idx], POINTER(MV_CC_DEVICE_INFO)).contents
        self.cam = MvCamera()
        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            QMessageBox.warning(self, "錯誤", "建立相機句柄失敗！")
            return

        ret = self.cam.MV_CC_OpenDevice()
        if ret != 0:
            QMessageBox.warning(self, "錯誤", "打開相機失敗！請確任網路與 IP。")
            return

        # 基本設置
        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        
        self.btn_open.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.btn_open.setText("相機已連線")

        # 取得當前曝光
        exp_val = MVCC_FLOATVALUE()
        self.cam.MV_CC_GetFloatValue("ExposureTime", exp_val)
        self.spin_exposure.setValue(int(exp_val.fCurValue))

        # 嘗試取得並設定當前硬體 FPS
        fps_val = MVCC_FLOATVALUE()
        ret_fps = self.cam.MV_CC_GetFloatValue("ResultingFrameRate", fps_val)
        if ret_fps == 0 and fps_val.fCurValue > 0:
            self.spin_fps.setValue(int(fps_val.fCurValue))

    def toggle_stream(self):
        if not self.is_streaming:
            ret = self.cam.MV_CC_StartGrabbing()
            if ret != 0:
                QMessageBox.warning(self, "錯誤", "無法開始讀取影像")
                return

            self.stream_thread = CameraThread(self.cam)
            self.stream_thread.set_resize_width(0)  # ROI 模式下不做額外縮放
            self.stream_thread.frame_ready.connect(self.update_ui)
            self.stream_thread.start()

            self.btn_start.setText("停止讀取")
            self.is_streaming = True
        else:
            if self.stream_thread:
                self.stream_thread.stop()
                self.stream_thread = None
            self.cam.MV_CC_StopGrabbing()
            self.btn_start.setText("開始讀取")
            self.is_streaming = False

    def auto_connect(self):
        """啟動後自動掃描並連線第一台相機，成功後立刻開始讀取"""
        self.scan_cameras()
        if self.device_list.nDeviceNum == 0:
            self.setWindowTitle("Single Camera Analysis — 未找到相機")
            return
        # 自動選第一台
        self.combo_cams.setCurrentIndex(0)
        self.open_camera()
        if self.cam is not None and self.btn_start.isEnabled():
            self.toggle_stream()

    def apply_roi(self):
        """停止取像 → 設定相機硬體 ROI → 重新啟動"""
        if not self.cam:
            return

        ox = self.spin_roi_x.value()
        oy = self.spin_roi_y.value()
        rw = self.spin_roi_w.value()
        rh = self.spin_roi_h.value()

        # 停止取像
        was_streaming = self.is_streaming
        if was_streaming:
            if self.stream_thread:
                self.stream_thread.stop()
                self.stream_thread = None
            self.cam.MV_CC_StopGrabbing()
            self.is_streaming = False

        # Hikvision SDK ROI 設定順序：先重置 Offset 為 0，再調寬高，再調 Offset
        self.cam.MV_CC_SetIntValue("OffsetX", 0)
        self.cam.MV_CC_SetIntValue("OffsetY", 0)
        self.cam.MV_CC_SetIntValue("Width", rw)
        self.cam.MV_CC_SetIntValue("Height", rh)
        self.cam.MV_CC_SetIntValue("OffsetX", ox)
        self.cam.MV_CC_SetIntValue("OffsetY", oy)

        # 更新讀取曝光値
        exp_val = MVCC_FLOATVALUE()
        self.cam.MV_CC_GetFloatValue("ExposureTime", exp_val)
        self.spin_exposure.setValue(int(exp_val.fCurValue))

        # 重新開始取像
        if was_streaming:
            ret = self.cam.MV_CC_StartGrabbing()
            if ret == 0:
                self.stream_thread = CameraThread(self.cam)
                self.stream_thread.set_resize_width(0)  # ROI 內不再軟體縮放
                self.stream_thread.frame_ready.connect(self.update_ui)
                self.stream_thread.start()
                self.is_streaming = True
                self.btn_start.setText("停止讀取")

    def reset_roi(self):
        """ROI 重置為相機最大感測區域"""
        if not self.cam:
            return

        was_streaming = self.is_streaming
        if was_streaming:
            if self.stream_thread:
                self.stream_thread.stop()
                self.stream_thread = None
            self.cam.MV_CC_StopGrabbing()
            self.is_streaming = False

        self.cam.MV_CC_SetIntValue("OffsetX", 0)
        self.cam.MV_CC_SetIntValue("OffsetY", 0)

        # 讀取相機支援的最大寬高
        max_w_val = MVCC_INTVALUE_EX()
        max_h_val = MVCC_INTVALUE_EX()
        self.cam.MV_CC_GetIntValueEx("WidthMax", max_w_val)
        self.cam.MV_CC_GetIntValueEx("HeightMax", max_h_val)
        max_w = max_w_val.nCurValue if max_w_val.nCurValue > 0 else 9999
        max_h = max_h_val.nCurValue if max_h_val.nCurValue > 0 else 9999

        self.cam.MV_CC_SetIntValue("Width",  int(max_w))
        self.cam.MV_CC_SetIntValue("Height", int(max_h))

        # 更新 UI
        self.spin_roi_x.setValue(0)
        self.spin_roi_y.setValue(0)
        self.spin_roi_w.setValue(int(max_w))
        self.spin_roi_h.setValue(int(max_h))

        if was_streaming:
            ret = self.cam.MV_CC_StartGrabbing()
            if ret == 0:
                self.stream_thread = CameraThread(self.cam)
                self.stream_thread.set_resize_width(0)
                self.stream_thread.frame_ready.connect(self.update_ui)
                self.stream_thread.start()
                self.is_streaming = True
                self.btn_start.setText("停止讀取")

    def on_resize_applied(self):
        pass   # 保留避免錯誤

    def on_scale_changed(self):
        pass   # slider 已移除，保留這裡避免錯誤

    def set_exposure(self):
        if self.cam:
            val = self.spin_exposure.value()
            self.cam.MV_CC_SetEnumValue("ExposureAuto", 0) # Off
            self.cam.MV_CC_SetFloatValue("ExposureTime", float(val))

    def set_fps(self):
        if self.cam:
            val = self.spin_fps.value()
            # 設定硬體 FPS
            self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
            self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(val))
            # 同步設定軟體取像上限
            if self.stream_thread:
                self.stream_thread.set_fps(val)

    def auto_white_balance(self):
        if self.cam:
            # 設定白平衡為 Once (2) 來觸發單次自動白平衡
            ret = self.cam.MV_CC_SetEnumValue("BalanceWhiteAuto", 2)
            if ret != 0:
                QMessageBox.warning(self, "錯誤", f"白平衡設定失敗 (錯誤碼: {hex(ret)})")

    def update_ui(self, bgr_img: np.ndarray, stats: dict):
        # 更新統計數據
        self.lbl_r.setText(f"R 平均: {stats['R']:.1f}")
        self.lbl_g.setText(f"G 平均: {stats['G']:.1f}")
        self.lbl_b.setText(f"B 平均: {stats['B']:.1f}")
        
        self.lbl_h.setText(f"H (Hue): {stats['H']:.1f}")
        self.lbl_s.setText(f"S (Sat): {stats['S']:.1f}")
        self.lbl_l.setText(f"L (Light): {stats['L']:.1f}")
        
        self.lbl_intensity.setText(f"平均光強: {stats['Intensity']:.2f}")
        self.lbl_exposure.setText(f"曝光值(us): {stats['Exposure']:.1f}")
        self.lbl_res.setText(f"當前解析度: {stats['Resolution']}")

        # 計算 FPS
        curr_time = time.time()
        if self._last_frame_time > 0:
            fps = 1.0 / max(curr_time - self._last_frame_time, 0.001)
            self.lbl_fps.setText(f"FPS: {fps:.1f}")
        self._last_frame_time = curr_time

        # 更新影像 (自適應 Label 大小保留比例)
        h, w, ch = bgr_img.shape
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        
        # 修正 PyQt QImage 記憶體參照問題
        qimg = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888).copy()
        
        pix = QPixmap.fromImage(qimg).scaled(
            self.image_label.width(), self.image_label.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pix)

    def update_hw_stats(self):
        try:
            self.lbl_cpu.setText(f"CPU: {psutil.cpu_percent():.1f}%")
            self.lbl_ram.setText(f"RAM: {psutil.virtual_memory().percent:.1f}%")
        except Exception as e:
            print("無法讀取硬體資訊:", e)

    def closeEvent(self, event):
        if self.is_streaming and self.stream_thread:
            self.stream_thread.stop()
        if self.cam:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SingleCamAnalysisApp()
    window.show()
    sys.exit(app.exec_())
