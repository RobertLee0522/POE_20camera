# -*- coding: utf-8 -*-
"""
white_balance.py
================
白平衡控制模組 - 調整 RGB 增益係數

提供的功能:
  - RGB 增益滑塊控制 (0.5 ~ 2.0)
  - 自動白平衡建議 (基於目前影像)
  - 增益預設組合 (日光、陰天、燈光等)
  - 即時預覽應用
"""

import numpy as np
from typing import Tuple, Optional
import cv2


class WhiteBalanceController:
    """白平衡控制引擎"""
    
    # 預設增益組合
    PRESETS = {
        'default': (1.0, 1.0, 1.0),           # 無調整
        'daylight': (1.0, 1.0, 1.1),          # 日光 (偏冷)
        'cloudy': (1.15, 1.0, 1.0),           # 陰天 (增加紅色)
        'tungsten': (1.5, 1.0, 0.8),          # 鎢絲燈 (暖色)
        'fluorescent': (0.9, 1.0, 1.2),       # 螢光燈 (冷色)
        'shade': (1.2, 1.0, 0.9),             # 陰影 (增加紅色)
    }
    
    def __init__(self):
        """初始化白平衡控制器"""
        self.r_gain = 1.0
        self.g_gain = 1.0
        self.b_gain = 1.0
        self.min_gain = 0.5
        self.max_gain = 2.0
    
    def set_gains(self, r_gain: float, g_gain: float, b_gain: float) -> None:
        """
        設定 RGB 增益係數
        
        Args:
            r_gain: 紅色增益 (0.5-2.0)
            g_gain: 綠色增益 (0.5-2.0)
            b_gain: 藍色增益 (0.5-2.0)
        """
        self.r_gain = max(self.min_gain, min(self.max_gain, r_gain))
        self.g_gain = max(self.min_gain, min(self.max_gain, g_gain))
        self.b_gain = max(self.min_gain, min(self.max_gain, b_gain))
    
    def apply_white_balance(self, frame: np.ndarray) -> np.ndarray:
        """
        將白平衡增益應用到影像
        
        Args:
            frame: BGR 格式影像
            
        Returns:
            調整後的影像 (uint8)
        """
        if frame is None or frame.size == 0:
            return frame
        
        # 轉換為浮點數以避免溢位
        adjusted = frame.astype(np.float32)
        
        # 應用增益 (記住 OpenCV 是 BGR 順序)
        adjusted[:, :, 0] = adjusted[:, :, 0] * self.b_gain  # Blue
        adjusted[:, :, 1] = adjusted[:, :, 1] * self.g_gain  # Green
        adjusted[:, :, 2] = adjusted[:, :, 2] * self.r_gain  # Red
        
        # 裁剪到有效範圍並轉換回 uint8
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return adjusted
    
    def set_preset(self, preset_name: str) -> bool:
        """
        應用預設白平衡組合
        
        Args:
            preset_name: 預設名稱 ('daylight', 'cloudy', 等)
            
        Returns:
            True 如果預設存在，False 否則
        """
        if preset_name not in self.PRESETS:
            return False
        
        r, g, b = self.PRESETS[preset_name]
        self.set_gains(r, g, b)
        return True
    
    def get_current_gains(self) -> Tuple[float, float, float]:
        """取得當前增益"""
        return (self.r_gain, self.g_gain, self.b_gain)
    
    def auto_white_balance(self, frame: np.ndarray) -> None:
        """
        自動白平衡調整 (灰卡算法)
        
        基本原理: 計算每個通道的平均值，使其相等
        假設影像中有一個灰色參考區域
        
        Args:
            frame: BGR 格式影像
        """
        if frame is None or frame.size == 0:
            return
        
        # 計算每個通道的平均值
        b_mean = frame[:, :, 0].mean()
        g_mean = frame[:, :, 1].mean()
        r_mean = frame[:, :, 2].mean()
        
        # 計算全局平均 (用於參考)
        global_mean = (b_mean + g_mean + r_mean) / 3.0
        
        # 計算增益 (使每個通道接近全局平均)
        if r_mean > 0:
            self.r_gain = global_mean / r_mean
        if g_mean > 0:
            self.g_gain = global_mean / g_mean
        if b_mean > 0:
            self.b_gain = global_mean / b_mean
        
        # 確保增益在有效範圍內
        self.r_gain = max(self.min_gain, min(self.max_gain, self.r_gain))
        self.g_gain = max(self.min_gain, min(self.max_gain, self.g_gain))
        self.b_gain = max(self.min_gain, min(self.max_gain, self.b_gain))
    
    def reset(self) -> None:
        """重置為預設值 (無調整)"""
        self.r_gain = 1.0
        self.g_gain = 1.0
        self.b_gain = 1.0
    
    @staticmethod
    def format_gains(r_gain: float, g_gain: float, b_gain: float) -> str:
        """格式化增益顯示"""
        return f"R: {r_gain:.2f} | G: {g_gain:.2f} | B: {b_gain:.2f}"
    
    @staticmethod
    def get_preset_names() -> list:
        """取得所有可用的預設名稱"""
        return list(WhiteBalanceController.PRESETS.keys())


# ============================================================
# 測試用例
# ============================================================
if __name__ == '__main__':
    # 建立測試影像
    test_frame = np.full((500, 500, 3), 100, dtype=np.uint8)
    
    wb = WhiteBalanceController()
    
    print("🎨 白平衡控制測試")
    print(f"\n初始增益: {WhiteBalanceController.format_gains(*wb.get_current_gains())}")
    
    # 測試預設
    for preset in wb.get_preset_names():
        wb.set_preset(preset)
        print(f"  {preset:12s}: {WhiteBalanceController.format_gains(*wb.get_current_gains())}")
    
    # 測試自動白平衡
    wb.reset()
    wb.auto_white_balance(test_frame)
    print(f"\n自動白平衡後: {WhiteBalanceController.format_gains(*wb.get_current_gains())}")
    
    # 測試應用效果
    wb.set_gains(1.5, 1.0, 0.8)
    adjusted = wb.apply_white_balance(test_frame)
    print(f"應用增益後: {adjusted.shape}, dtype={adjusted.dtype}")
