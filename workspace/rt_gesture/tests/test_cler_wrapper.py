from __future__ import annotations

import numpy as np
import pandas as pd

from rt_gesture.cler import compute_cler, detect_gesture_events, map_gestures_to_probabilities


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

