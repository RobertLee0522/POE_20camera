# -*- coding: utf-8 -*-
"""
Shared pytest configuration.

The unit suite deliberately covers only modules that need neither Aravis
nor Qt, so it runs on a plain CI box with numpy + OpenCV installed.
"""

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
