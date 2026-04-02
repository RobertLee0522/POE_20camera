# -*- coding: utf-8 -*-
"""
camera_controller.py
====================
簡化的單一相機控制模組 - 基於 Hikvision MVS SDK

提供的功能:
  - 相機檢測與連接
  - 取像開始/停止
  - 基本參數調整 (曝光、增益)
  - 幀捕獲
"""

import sys
import os
import platform
import threading
import numpy as np
import time
from typing import Optional, List, Dict, Callable

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
    HIKVISION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  提示: Hikvision MVS SDK 未安裝")
    HIKVISION_AVAILABLE = False


class CameraController:
    """單一相機控制器"""
    
    def __init__(self, device_index: int = 0):
        """
        初始化相機控制器
        
        Args:
            device_index: 設備索引 (第一台相機為 0)
        """
        self.device_index = device_index
        self.obj_cam = MvCamera()
        self.st_device_list = MV_CC_DEVICE_INFO_LIST()
        self.is_connected = False
        self.is_grabbing = False
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self.frame_info = MV_FRAME_OUT_INFO_EX()
        self.exposure_time = 0.0
        self.gain = 0.0
        self.camera_name = "Unknown"
        self.camera_ip = "Unknown"
    
    def enum_devices(self) -> List[Dict]:
        """
        列舉可用的相機設備
        
        Returns:
            設備信息列表，每項包含:
            {
                'index': 設備索引,
                'name': 相機型號,
                'ip': IP 地址 (如果有)
            }
        """
        if not HIKVISION_AVAILABLE:
            print("❌ Hikvision SDK 不可用")
            return []
        
        devices = []
        
        # 列舉設備
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, self.st_device_list)
        if ret != 0:
            print(f"❌ 列舉設備失敗: {ret}")
            return []
        
        # 取得設備信息
        for i in range(self.st_device_list.nDeviceNum):
            mvcc_dev_info = self.st_device_list.pDeviceInfo[i]
            if mvcc_dev_info is None:
                continue
            
            device_info = {
                'index': i,
                'name': '',
                'ip': ''
            }
            
            # GigE 相機
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                nIp1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp) & 0xff000000) >> 24
                nIp2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp) & 0x00ff0000) >> 16
                nIp3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp) & 0x0000ff00) >> 8
                nIp4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp) & 0x000000ff
                device_info['ip'] = f"{nIp1}.{nIp2}.{nIp3}.{nIp4}"
            
            # USB 相機
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                device_info['ip'] = "USB"
            
            devices.append(device_info)
        
        return devices
    
    def connect(self, device_index: int = 0) -> bool:
        """
        連接到指定的相機
        
        Args:
            device_index: 設備索引
            
        Returns:
            True 連接成功，False 失敗
        """
        if not HIKVISION_AVAILABLE:
            print("❌ Hikvision SDK 不可用")
            return False
        
        if self.is_connected:
            print("⚠️  已經連接到相機，先斷開連接")
            return False
        
        # 列舉設備
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, self.st_device_list)
        if ret != 0 or device_index >= self.st_device_list.nDeviceNum:
            print(f"❌ 無效的設備索引: {device_index}")
            return False
        
        # 創建相機對象
        self.obj_cam = MvCamera()
        
        # 連接設備
        ret = self.obj_cam.MV_CC_CreateHandle(self.st_device_list.pDeviceInfo[device_index])
        if ret != 0:
            print(f"❌ 創建句柄失敗: {ret}")
            return False
        
        ret = self.obj_cam.MV_CC_OpenDevice(MV_ACCESS_EXCLUSIVE)
        if ret != 0:
            print(f"❌ 打開設備失敗: {ret}")
            return False
        
        self.is_connected = True
        self.device_index = device_index
        
        print(f"✅ 已連接到相機 #{device_index}")
        return True
    
    def disconnect(self) -> bool:
        """斷開相機連接"""
        if not self.is_connected:
            return True
        
        if self.is_grabbing:
            self.stop_grabbing()
        
        ret = self.obj_cam.MV_CC_CloseDevice()
        if ret != 0:
            print(f"❌ 關閉設備失敗: {ret}")
            return False
        
        self.is_connected = False
        print("✅ 已斷開相機連接")
        return True
    
    def start_grabbing(self) -> bool:
        """開始取像"""
        if not self.is_connected:
            print("❌ 未連接到相機")
            return False
        
        if self.is_grabbing:
            return True
        
        ret = self.obj_cam.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"❌ 開始取像失敗: {ret}")
            return False
        
        self.is_grabbing = True
        print("✅ 已開始取像")
        return True
    
    def stop_grabbing(self) -> bool:
        """停止取像"""
        if not self.is_grabbing:
            return True
        
        ret = self.obj_cam.MV_CC_StopGrabbing()
        if ret != 0:
            print(f"❌ 停止取像失敗: {ret}")
            return False
        
        self.is_grabbing = False
        print("✅ 已停止取像")
        return True
    
    def get_frame(self, timeout_ms: int = 100) -> Optional[np.ndarray]:
        """
        取得單幀影像
        
        Args:
            timeout_ms: 超時時間 (毫秒)
            
        Returns:
            BGR 格式的 numpy 陣列，或 None 如果失敗
        """
        if not self.is_grabbing:
            return None
        
        try:
            ret = self.obj_cam.MV_CC_GetOneFrameTimeout(
                self.frame_info,
                timeout_ms
            )
            
            if ret != 0:
                return None
            
            # 轉換為 numpy 陣列
            frame_data = np.asarray(self.frame_info.pBufAddr)
            
            if self.frame_info.enPixelType == PixelType_Gvsp_Mono8:
                # 單色圖像
                frame = frame_data.reshape((
                    self.frame_info.nHeight,
                    self.frame_info.nWidth
                ))
                # 轉換為 BGR (複製到三個通道)
                frame = np.stack([frame, frame, frame], axis=2)
            
            elif self.frame_info.enPixelType == PixelType_Gvsp_RGB8:
                # RGB 圖像
                frame = frame_data.reshape((
                    self.frame_info.nHeight,
                    self.frame_info.nWidth,
                    3
                ))
                # 轉換為 BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            else:
                # Bayer 或其他格式 - 需要轉換
                # 這裡簡化處理，可根據具體情況擴展
                frame = frame_data.reshape((
                    self.frame_info.nHeight,
                    self.frame_info.nWidth,
                    3
                ))
            
            frame = frame.astype(np.uint8)
            
            with self.frame_lock:
                self.current_frame = frame.copy()
            
            return frame
            
        except Exception as e:
            print(f"❌ 取幀失敗: {e}")
            return None
    
    def set_exposure(self, exposure_time_us: float) -> bool:
        """
        設定曝光時間
        
        Args:
            exposure_time_us: 曝光時間 (微秒)
            
        Returns:
            True 成功，False 失敗
        """
        if not self.is_connected:
            return False
        
        # 設定為自動曝光先關閉
        ret = self.obj_cam.MV_CC_SetBoolValue(MV_CAM_PARAM_AUTO_EXPOSURE, False)
        
        ret = self.obj_cam.MV_CC_SetFloatValue(
            MV_CAM_PARAM_EXPOSURE_TIME,
            exposure_time_us
        )
        
        if ret == 0:
            self.exposure_time = exposure_time_us
            return True
        return False
    
    def set_gain(self, gain_db: float) -> bool:
        """
        設定增益
        
        Args:
            gain_db: 增益 (dB)
            
        Returns:
            True 成功，False 失敗
        """
        if not self.is_connected:
            return False
        
        ret = self.obj_cam.MV_CC_SetFloatValue(MV_CAM_PARAM_GAIN, gain_db)
        
        if ret == 0:
            self.gain = gain_db
            return True
        return False
    
    def get_status(self) -> Dict:
        """取得相機狀態"""
        return {
            'connected': self.is_connected,
            'grabbing': self.is_grabbing,
            'device_index': self.device_index,
            'exposure_time': self.exposure_time,
            'gain': self.gain,
        }


# 為了相容性
import cv2

if __name__ == '__main__':
    print("🎥 相機控制模組已載入")
