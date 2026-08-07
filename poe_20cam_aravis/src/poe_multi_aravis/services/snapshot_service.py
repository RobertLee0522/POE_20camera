# -*- coding: utf-8 -*-
"""
services/snapshot_service.py
==============================
Saves snapshots from the current frame.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


class SnapshotService:
    """Saves a BGR numpy frame as an image file."""

    def __init__(self, save_dir: str = "./snapshots") -> None:
        self._save_dir = save_dir

    def save(
        self,
        frame: np.ndarray,
        filename: Optional[str] = None,
        ext: str = ".png",
    ) -> str:
        """
        Save frame to disk.

        Args:
            frame:    BGR uint8 ndarray.
            filename: Base filename without extension.  If None, uses timestamp.
            ext:      Extension – .png, .jpg, or .bmp.

        Returns:
            Absolute path of saved file.

        Raises:
            ValueError: Unsupported extension.
            IOError:    Write failure.
        """
        if ext.lower() not in SUPPORTED_EXTS:
            raise ValueError(f"Unsupported format: {ext}")

        os.makedirs(self._save_dir, exist_ok=True)

        if filename is None:
            filename = datetime.now().strftime("snapshot_%Y%m%d_%H%M%S%f")

        path = os.path.join(self._save_dir, filename + ext)
        ok = cv2.imwrite(path, frame)
        if not ok:
            raise IOError(f"cv2.imwrite failed for: {path}")

        log.info("Snapshot saved: %s", path)
        return path
