# -*- coding: utf-8 -*-
"""
image_analyzer.py
=================
影像分析模組 - 計算 RGB、HSL、光強度等統計資訊

提供的功能:
  - RGB 通道平均值計算
  - HSL 色彩空間轉換與分析
  - 平均光強度計算 (ITU-R BT.601 加權)
  - 曝光統計資訊
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional


class ImageAnalyzer:
    """影像分析引擎"""
    
    def __init__(self):
        """初始化分析器"""
        self.last_rgb_avg = (0, 0, 0)      # B, G, R
        self.last_hsl_avg = (0, 0, 0)      # H, S, L
        self.last_brightness = 0.0         # 0-255
        self.exposure_time = 0.0           # 微秒
        
    def analyze_frame(self, frame: np.ndarray, exposure_time: Optional[float] = None) -> Dict:
        """
        分析單一影像幀
        
        Args:
            frame: BGR 格式的 numpy 陣列 (OpenCV 標準)
            exposure_time: 曝光時間 (微秒)，可選
            
        Returns:
            字典包含以下鍵:
            {
                'rgb_avg': (R, G, B) 平均值 (0-255),
                'hsl_avg': (H, S, L) 平均值,
                'brightness': 平均光強度 (0-255),
                'exposure_time': 曝光時間 (微秒),
                'frame_shape': (height, width, channels)
            }
        """
        if frame is None or frame.size == 0:
            return self._empty_result()
        
        # 1. 計算 RGB 平均值
        rgb_avg = self._calculate_rgb_average(frame)
        
        # 2. 計算 HSL 平均值
        hsl_avg = self._calculate_hsl_average(frame)
        
        # 3. 計算平均光強度
        brightness = self._calculate_brightness(frame)
        
        # 4. 保存結果用於後續查詢
        self.last_rgb_avg = rgb_avg
        self.last_hsl_avg = hsl_avg
        self.last_brightness = brightness
        if exposure_time is not None:
            self.exposure_time = exposure_time
        
        return {
            'rgb_avg': rgb_avg,
            'hsl_avg': hsl_avg,
            'brightness': brightness,
            'exposure_time': self.exposure_time,
            'frame_shape': frame.shape
        }
    
    def _calculate_rgb_average(self, frame: np.ndarray) -> Tuple[float, float, float]:
        """
        計算 RGB 三通道平均值
        
        Args:
            frame: BGR 格式影像
            
        Returns:
            (R_avg, G_avg, B_avg) 其中值為 0-255
        """
        # OpenCV 使用 BGR，所以需要反轉到 RGB
        b_avg = float(frame[:, :, 0].mean())
        g_avg = float(frame[:, :, 1].mean())
        r_avg = float(frame[:, :, 2].mean())
        
        return (r_avg, g_avg, b_avg)
    
    def _calculate_hsl_average(self, frame: np.ndarray) -> Tuple[float, float, float]:
        """
        計算 HSL 色彩空間平均值
        
        算法:
        1. 轉換 BGR → RGB
        2. 將 RGB 正規化到 [0, 1]
        3. 計算 H, S, L
        4. 返回平均值 (H: 0-180, S: 0-100, L: 0-100)
        
        Args:
            frame: BGR 格式影像
            
        Returns:
            (H_avg, S_avg, L_avg)
            - H: 0-180 度 (OpenCV Hue 範圍)
            - S: 0-100 百分比
            - L: 0-100 百分比
        """
        # 轉換 BGR → HSV (OpenCV 只支援 HSV)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        h_avg = float(hsv[:, :, 0].mean())  # 0-180
        s_avg = float(hsv[:, :, 1].mean())  # 0-255
        v_avg = float(hsv[:, :, 2].mean())  # 0-255
        
        # 將 S, V 轉換為百分比 (0-100)
        s_percent = (s_avg / 255.0) * 100.0
        l_percent = (v_avg / 255.0) * 100.0
        
        return (h_avg, s_percent, l_percent)
    
    def _calculate_brightness(self, frame: np.ndarray) -> float:
        """
        計算平均光強度 (亮度)
        
        使用 ITU-R BT.601 標準公式:
            Y = 0.299*R + 0.587*G + 0.114*B
        
        Args:
            frame: BGR 格式影像
            
        Returns:
            平均光強度 (0-255)
        """
        # 分離通道
        b = frame[:, :, 0].astype(np.float32)
        g = frame[:, :, 1].astype(np.float32)
        r = frame[:, :, 2].astype(np.float32)
        
        # ITU-R BT.601 加權平均
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        brightness = float(gray.mean())
        
        return brightness
    
    def _empty_result(self) -> Dict:
        """返回空結果"""
        return {
            'rgb_avg': (0, 0, 0),
            'hsl_avg': (0, 0, 0),
            'brightness': 0.0,
            'exposure_time': 0.0,
            'frame_shape': (0, 0, 0)
        }
    
    def get_last_stats(self) -> Dict:
        """取得上一幀的統計資訊"""
        return {
            'rgb_avg': self.last_rgb_avg,
            'hsl_avg': self.last_hsl_avg,
            'brightness': self.last_brightness,
            'exposure_time': self.exposure_time
        }
    
    @staticmethod
    def format_rgb(rgb_tuple: Tuple[float, float, float]) -> str:
        """格式化 RGB 顯示"""
        r, g, b = rgb_tuple
        return f"R: {r:.1f} | G: {g:.1f} | B: {b:.1f}"
    
    @staticmethod
    def format_hsl(hsl_tuple: Tuple[float, float, float]) -> str:
        """格式化 HSL 顯示"""
        h, s, l = hsl_tuple
        return f"H: {h:.1f}° | S: {s:.1f}% | L: {l:.1f}%"
    
    @staticmethod
    def format_brightness(brightness: float) -> str:
        """格式化光強度顯示"""
        percent = (brightness / 255.0) * 100.0
        return f"Brightness: {brightness:.1f}/255 ({percent:.1f}%)"


# ============================================================
# 測試用例
# ============================================================
if __name__ == '__main__':
    # 建立測試影像 (500x500, 隨機顏色)
    test_frame = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
    
    analyzer = ImageAnalyzer()
    result = analyzer.analyze_frame(test_frame, exposure_time=10000)
    
    print("🔍 影像分析結果:")
    print(f"  RGB: {ImageAnalyzer.format_rgb(result['rgb_avg'])}")
    print(f"  HSL: {ImageAnalyzer.format_hsl(result['hsl_avg'])}")
    print(f"  {ImageAnalyzer.format_brightness(result['brightness'])}")
    print(f"  曝光時間: {result['exposure_time']} μs")
    print(f"  影像尺寸: {result['frame_shape']}")
