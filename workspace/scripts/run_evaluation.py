#!/usr/bin/env python3
"""Run RT-Gesture CLER evaluation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rt_gesture.evaluate import main


if __name__ == "__main__":
    raise SystemExit(main())

