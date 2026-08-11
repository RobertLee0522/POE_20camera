# -*- coding: utf-8 -*-
"""
domain/errors.py
================
Application-specific error hierarchy.
"""


class PoeError(Exception):
    """Base for all application errors."""


class CameraNotFoundError(PoeError):
    """No camera with the given ID was discovered."""


class CameraConnectionError(PoeError):
    """Could not connect to the camera."""


class AcquisitionError(PoeError):
    """Error during stream setup or buffer management."""


class FeatureError(PoeError):
    """A GenICam feature read/write failed."""


class FeatureUnavailableError(FeatureError):
    """The camera does not expose the requested feature."""


class PixelFormatError(PoeError):
    """The pixel format is unsupported or incompatible."""


class ConfigurationError(PoeError):
    """Invalid application configuration."""


class GpuError(PoeError):
    """GPU/CUDA failure – caller should fall back to CPU."""


class LoggingError(PoeError):
    """Non-fatal coverage log write failure."""
