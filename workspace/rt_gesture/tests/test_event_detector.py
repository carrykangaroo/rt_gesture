from __future__ import annotations

import numpy as np

from rt_gesture.event_detector import EventDetector, EventDetectorConfig


def test_rest_state_suppression() -> None:
    detector = EventDetector()
    events = detector.process_frame(np.full(9, 0.1), timestamp=0.0)
    assert events == []


def test_threshold_crossing_produces_event() -> None:
    detector = EventDetector()
    detector.process_frame(np.zeros(9), timestamp=0.0)
    probs = np.zeros(9)
    probs[0] = 0.8
    events = detector.process_frame(probs, timestamp=0.1)
    assert len(events) == 1
    assert events[0].gesture == "index_press"
    assert events[0].confidence == 0.8


def test_rejection_threshold_filters_low_confidence() -> None:
    detector = EventDetector(EventDetectorConfig(rejection_threshold=0.9))
    detector.process_frame(np.zeros(9), timestamp=0.0)
    probs = np.zeros(9)
    probs[4] = 0.7
    events = detector.process_frame(probs, timestamp=0.1)
    assert events == []


def test_debounce_blocks_quick_retrigger() -> None:
    detector = EventDetector(EventDetectorConfig(debounce_sec=0.05))

    first = np.zeros(9)
    first[0] = 0.9
    detector.process_frame(first, timestamp=1.0)

    low = np.zeros(9)
    detector.process_frame(low, timestamp=1.01)

    second = np.zeros(9)
    second[0] = 0.95
    blocked = detector.process_frame(second, timestamp=1.03)
    assert blocked == []

    detector.process_frame(low, timestamp=1.06)
    accepted = detector.process_frame(second, timestamp=1.11)
    assert len(accepted) == 1


def test_reset_clears_internal_state() -> None:
    detector = EventDetector()
    high = np.zeros(9)
    high[2] = 0.95
    detector.process_frame(high, timestamp=0.1)
    detector.reset()
    events = detector.process_frame(high, timestamp=0.2)
    assert len(events) == 1
    assert events[0].gesture == "middle_press"


def test_process_batch_accepts_channel_major() -> None:
    detector = EventDetector()
    probs = np.zeros((9, 3))
    probs[1, 1] = 0.8
    timestamps = np.array([0.0, 0.1, 0.2])
    events = detector.process_batch(probs, timestamps)
    assert len(events) == 1
    assert events[0].gesture == "index_release"

