"""Streaming gesture event detection and post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from rt_gesture.constants import (
    DEFAULT_DEBOUNCE_SEC,
    DEFAULT_REJECTION_THRESHOLD,
    DEFAULT_REST_THRESHOLD,
    DEFAULT_THRESHOLD,
    GESTURE_NAMES,
    GestureType,
)


class GestureEvent(NamedTuple):
    gesture: str
    timestamp: float
    confidence: float


@dataclass
class EventDetectorConfig:
    threshold: float = DEFAULT_THRESHOLD
    rejection_threshold: float = DEFAULT_REJECTION_THRESHOLD
    rest_threshold: float = DEFAULT_REST_THRESHOLD
    debounce_sec: float = DEFAULT_DEBOUNCE_SEC


class EventDetector:
    """Stateful per-frame event detector for streaming probabilities."""

    def __init__(self, config: EventDetectorConfig | None = None) -> None:
        self.cfg = config or EventDetectorConfig()
        self._prev_probs: np.ndarray | None = None
        self._last_event_time: dict[str, float] = {}

    def reset(self) -> None:
        self._prev_probs = None
        self._last_event_time.clear()

    def process_frame(self, probs: np.ndarray, timestamp: float) -> list[GestureEvent]:
        if probs.shape != (len(GestureType),):
            raise ValueError(f"Expected shape (9,), got {probs.shape}")

        events: list[GestureEvent] = []
        if np.all(probs < self.cfg.rest_threshold):
            self._prev_probs = probs.copy()
            return events

        if self._prev_probs is None:
            crossings = probs >= self.cfg.threshold
        else:
            crossings = (probs >= self.cfg.threshold) & (self._prev_probs < self.cfg.threshold)

        for channel_idx in np.where(crossings)[0]:
            gesture_name = GESTURE_NAMES[channel_idx]
            confidence = float(probs[channel_idx])
            if confidence < self.cfg.rejection_threshold:
                continue

            last_timestamp = self._last_event_time.get(gesture_name, -np.inf)
            if timestamp - last_timestamp < self.cfg.debounce_sec:
                continue

            events.append(
                GestureEvent(
                    gesture=gesture_name,
                    timestamp=float(timestamp),
                    confidence=confidence,
                )
            )
            self._last_event_time[gesture_name] = float(timestamp)

        self._prev_probs = probs.copy()
        return events

    def process_batch(
        self,
        probs_batch: np.ndarray,
        timestamps: np.ndarray,
    ) -> list[GestureEvent]:
        if probs_batch.ndim != 2:
            raise ValueError(f"Expected 2D probs_batch, got {probs_batch.ndim}D")
        if timestamps.ndim != 1:
            raise ValueError(f"Expected 1D timestamps, got {timestamps.ndim}D")

        if probs_batch.shape[0] == len(GestureType):
            probs_time_major = probs_batch.T
        elif probs_batch.shape[1] == len(GestureType):
            probs_time_major = probs_batch
        else:
            raise ValueError(f"Expected one probs dimension to be 9, got {probs_batch.shape}")

        if probs_time_major.shape[0] != timestamps.shape[0]:
            raise ValueError("time length mismatch between probs and timestamps")

        all_events: list[GestureEvent] = []
        for idx in range(probs_time_major.shape[0]):
            all_events.extend(
                self.process_frame(
                    probs=probs_time_major[idx],
                    timestamp=float(timestamps[idx]),
                )
            )
        return all_events

