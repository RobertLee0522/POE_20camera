# -*- coding: utf-8 -*-
"""
frame_resizer.py
================
畫面縮放模組 - 調整影像顯示尺寸

提供的功能:
  - 預設縮放比例 (100%, 75%, 50%, 25%)
  - 自訂寬度和高度
  - 保持長寬比
  - 高品質插值算法
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class FrameResizer:
    """畫面縮放引擎"""
    
    # 預設縮放比例
    PRESETS = {
        '100%': 1.0,
        '75%': 0.75,
        '50%': 0.5,
        '25%': 0.25,
    }
    
    # 插值算法
    INTERPOLATION_METHODS = {
        'linear': cv2.INTER_LINEAR,           # 雙線性 (推薦，預設)
        'nearest': cv2.INTER_NEAREST,         # 最近鄰 (最快)
        'cubic': cv2.INTER_CUBIC,             # 雙三次 (最高品質)
        'lanczos4': cv2.INTER_LANCZOS4,       # 高級 (緩慢但品質最好)
    }
    
    def __init__(self, original_shape: Optional[Tuple[int, int]] = None):
        """
        初始化縮放器
        
        Args:
            original_shape: 原始影像尺寸 (height, width)
        """
        self.original_height = original_shape[0] if original_shape else None
        self.original_width = original_shape[1] if original_shape else None
        self.scale_ratio = 1.0
        self.interpolation = cv2.INTER_LINEAR
        self.keep_aspect_ratio = True
    
    def update_original_shape(self, frame: np.ndarray) -> None:
        """更新原始影像尺寸"""
        if frame is not None and frame.size > 0:
            self.original_height, self.original_width = frame.shape[:2]
    
    def set_scale_ratio(self, ratio: float) -> None:
        """
        設定縮放比例
        
        Args:
            ratio: 縮放比例 (0.1 ~ 2.0)
        """
        self.scale_ratio = max(0.1, min(2.0, ratio))
    
    def set_target_size(self, width: int, height: int) -> None:
        """
        設定目標尺寸 (精確值，不保持比例)
        
        Args:
            width: 目標寬度 (像素)
            height: 目標高度 (像素)
        """
        self.target_width = max(10, width)
        self.target_height = max(10, height)
        self.scale_ratio = None  # 使用精確尺寸而非比例
    
    def set_target_width(self, width: int) -> None:
        """
        設定目標寬度 (保持長寬比)
        
        Args:
            width: 目標寬度 (像素)
        """
        if self.original_width and self.original_height:
            aspect_ratio = self.original_height / self.original_width
            self.target_width = width
            self.target_height = int(width * aspect_ratio)
    
    def set_interpolation(self, method: str) -> bool:
        """
        設定插值方法
        
        Args:
            method: 方法名稱 ('linear', 'nearest', 'cubic', 'lanczos4')
            
        Returns:
            True 如果方法有效
        """
        if method not in self.INTERPOLATION_METHODS:
            return False
        
        self.interpolation = self.INTERPOLATION_METHODS[method]
        return True
    
    def set_preset(self, preset_name: str) -> bool:
        """
        應用預設縮放比例
        
        Args:
            preset_name: 預設名稱 ('100%', '75%', '50%', '25%')
            
        Returns:
            True 如果預設存在
        """
        if preset_name not in self.PRESETS:
            return False
        
        self.scale_ratio = self.PRESETS[preset_name]
        return True
    
    def resize(self, frame: np.ndarray) -> np.ndarray:
        """
        縮放影像
        
        Args:
            frame: 輸入影像
            
        Returns:
            縮放後的影像
        """
        if frame is None or frame.size == 0:
            return frame
        
        # 更新原始尺寸 (如果需要)
        if self.original_width is None:
            self.update_original_shape(frame)
        
        height, width = frame.shape[:2]
        
        # 計算目標尺寸
        if self.scale_ratio is not None:
            new_width = int(width * self.scale_ratio)
            new_height = int(height * self.scale_ratio)
        else:
            # 使用精確尺寸
            new_width = self.target_width
            new_height = self.target_height
        
        # 避免過小尺寸
        new_width = max(10, new_width)
        new_height = max(10, new_height)
        
        # 執行縮放
        resized = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=self.interpolation
        )
        
        return resized
    
    def get_current_size(self) -> Tuple[int, int]:
        """取得當前設定的目標尺寸 (寬, 高)"""
        if self.original_width and self.original_height:
            width = int(self.original_width * self.scale_ratio)
            height = int(self.original_height * self.scale_ratio)
            return (width, height)
        return (0, 0)
    
    def reset(self) -> None:
        """重置為原始尺寸 (100%)"""
        self.scale_ratio = 1.0
    
    @staticmethod
    def get_preset_names() -> list:
        """取得所有可用的預設名稱"""
        return list(FrameResizer.PRESETS.keys())
    
    @staticmethod
    def get_interpolation_methods() -> list:
        """取得所有可用的插值方法"""
        return list(FrameResizer.INTERPOLATION_METHODS.keys())


# ============================================================
# 測試用例
# ============================================================
if __name__ == '__main__':
    # 建立測試影像 (1080x720)
    test_frame = np.random.randint(0, 256, (720, 1080, 3), dtype=np.uint8)
    
    resizer = FrameResizer(test_frame.shape[:2])
    
    print("📐 畫面縮放測試")
    print(f"原始尺寸: {test_frame.shape[:2]} (height, width)")
    
    # 測試預設
    for preset in resizer.get_preset_names():
        resizer.set_preset(preset)
        w, h = resizer.get_current_size()
        print(f"  {preset:5s}: {h:4d}x{w:4d}")
    
    # 測試縮放
    resizer.set_preset('50%')
    resized = resizer.resize(test_frame)
    print(f"\n應用 50% 縮放後: {resized.shape[:2]}")
    
    # 測試自訂寬度
    resizer.set_target_width(640)
    w, h = resizer.get_current_size()
    print(f"設定目標寬度 640 後: {h}x{w}")
