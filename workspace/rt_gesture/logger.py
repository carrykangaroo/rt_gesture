"""Logging helpers and lightweight runtime stores."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


def setup_logging(log_dir: str | Path, log_level: str = "INFO") -> Path:
    """Configure root logging and return the run directory."""
    run_dir = Path(log_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Tests may call setup repeatedly, clear old handlers to avoid duplicate logs.
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(run_dir / "runtime.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    return run_dir


@dataclass(frozen=True)
class GestureEventRecord:
    gesture: str
    timestamp: float
    confidence: float


class EventLogger:
    """Append gesture events to a JSON Lines file."""

    def __init__(self, run_dir: str | Path) -> None:
        self._path = Path(run_dir) / "events.jsonl"
        self._file = self._path.open("w", encoding="utf-8")

    def log_event(self, gesture: str, timestamp: float, confidence: float) -> None:
        payload = {
            "gesture": gesture,
            "timestamp": float(timestamp),
            "confidence": round(float(confidence), 6),
            "wall_time": datetime.now().isoformat(),
        }
        self._file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class PredictionStore:
    """Buffer frame probabilities and persist them as a compressed NPZ."""

    def __init__(self, run_dir: str | Path) -> None:
        self._run_dir = Path(run_dir)
        self._prob_chunks: list[np.ndarray] = []
        self._time_chunks: list[np.ndarray] = []

    def append(self, probs: np.ndarray, times: np.ndarray) -> None:
        if probs.ndim != 2:
            raise ValueError(f"expected probs with 2 dims, got {probs.ndim}")
        if times.ndim != 1:
            raise ValueError(f"expected times with 1 dim, got {times.ndim}")
        if probs.shape[1] != times.shape[0]:
            raise ValueError("probs time dimension must match times length")
        self._prob_chunks.append(np.asarray(probs))
        self._time_chunks.append(np.asarray(times))

    def save(self) -> Path | None:
        if not self._prob_chunks:
            return None
        path = self._run_dir / "predictions.npz"
        all_probs = np.concatenate(self._prob_chunks, axis=1)
        all_times = np.concatenate(self._time_chunks, axis=0)
        np.savez_compressed(path, probs=all_probs, times=all_times)
        return path

