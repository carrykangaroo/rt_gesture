from __future__ import annotations

import numpy as np
import pandas as pd

from rt_gesture.cler import (
    compute_cler,
    debounce_events,
    detect_gesture_events,
    get_matched_indices,
    map_gestures_to_probabilities,
)


def test_cler_wrapper_basic_interfaces() -> None:
    probs = np.zeros((9, 20), dtype=np.float32)
    times = np.arange(20, dtype=np.float64) * 0.005
    probs[0, 5:9] = 0.9
    probs[1, 12:14] = 0.95

    mapped = map_gestures_to_probabilities(probs, times)
    events = detect_gesture_events(mapped, threshold=0.35, debounce=0.05)
    assert "time" in mapped
    assert set(events.columns) == {"name", "time", "start", "end"}

    prompts = pd.DataFrame(
        [
            {"name": "index_press", "time": float(times[5]), "start": float(times[5]), "end": float(times[5])},
            {"name": "index_release", "time": float(times[12]), "start": float(times[12]), "end": float(times[12])},
        ]
    )
    cler = compute_cler(probs, times, prompts)
    assert isinstance(cler, float)
    assert 0.0 <= cler <= 1.0


def test_debounce_events_rules_for_press_and_release() -> None:
    events = [
        ("index_press", 1.000),
        ("middle_press", 1.010),
        ("index_release", 1.020),
        ("index_release", 1.030),
        ("middle_release", 1.035),
    ]

    debounced = debounce_events(events, debounce=0.05)

    assert debounced == [
        ("index_press", 1.000),
        ("index_release", 1.020),
        ("middle_release", 1.035),
    ]


def test_detect_gesture_events_counts_first_timestep_crossing() -> None:
    times = np.arange(10, dtype=np.float64) * 0.005
    probs = np.zeros((9, 10), dtype=np.float32)
    probs[0, 0] = 0.9

    mapped = map_gestures_to_probabilities(probs, times)
    events = detect_gesture_events(mapped, threshold=0.35, debounce=0.05)

    assert len(events) == 1
    assert events.iloc[0]["name"] == "index_press"
    assert events.iloc[0]["time"] == times[0]


def test_compute_cler_zero_for_perfect_alignment() -> None:
    probs = np.zeros((9, 120), dtype=np.float32)
    times = np.arange(120, dtype=np.float64) * 0.005
    probs[0, 20:30] = 0.95
    probs[1, 60:70] = 0.9
    prompts = pd.DataFrame(
        [
            {"name": "index_press", "time": float(times[20]), "start": float(times[20]), "end": float(times[20])},
            {"name": "index_release", "time": float(times[60]), "start": float(times[60]), "end": float(times[60])},
        ]
    )

    cler = compute_cler(probs, times, prompts)

    assert cler == 0.0


def test_compute_cler_nonzero_for_label_mismatch() -> None:
    probs = np.zeros((9, 120), dtype=np.float32)
    times = np.arange(120, dtype=np.float64) * 0.005
    probs[2, 20:30] = 0.95
    probs[2, 60:70] = 0.95
    prompts = pd.DataFrame(
        [
            {
                "name": "index_press",
                "time": float(times[20]),
                "start": float(times[20]),
                "end": float(times[20]),
            },
            {
                "name": "middle_press",
                "time": float(times[60]),
                "start": float(times[60]),
                "end": float(times[60]),
            },
        ]
    )

    cler = compute_cler(probs, times, prompts)

    assert cler > 0.0


def test_get_matched_indices_alignment_output_shape() -> None:
    left = pd.DataFrame(
        [
            {"name": "index_press", "time": 1.0, "start": 0.95, "end": 1.05},
            {"name": "index_release", "time": 2.0, "start": 1.95, "end": 2.05},
        ]
    )
    right = pd.DataFrame(
        [
            {"name": "index_press", "time": 1.01, "start": 0.96, "end": 1.06},
            {"name": "index_release", "time": 2.01, "start": 1.96, "end": 2.06},
        ]
    )

    matches = get_matched_indices(left, right)

    assert len(matches) == 2
    assert matches[0] == (0, 0)
    assert matches[1] == (1, 1)
