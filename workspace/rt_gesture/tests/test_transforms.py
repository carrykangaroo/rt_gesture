from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from rt_gesture.constants import GestureType
from rt_gesture.transforms import DiscreteGesturesTransform, RotationAugmentation


def _build_timeseries(num_steps: int = 1000) -> np.ndarray:
    dtype = np.dtype([("emg", np.float32, (16,)), ("time", np.float64)])
    timeseries = np.zeros(num_steps, dtype=dtype)
    timeseries["emg"] = np.random.randn(num_steps, 16).astype(np.float32)
    timeseries["time"] = np.linspace(0.0, 0.999, num_steps, dtype=np.float64)
    return timeseries


def test_discrete_gestures_transform_output_shapes() -> None:
    timeseries = _build_timeseries(1000)
    prompts = pd.DataFrame(
        [
            {"name": GestureType.index_press.name, "time": 0.5},
            {"name": GestureType.middle_press.name, "time": 0.7},
            {"name": "unknown", "time": 0.8},
        ]
    )
    transform = DiscreteGesturesTransform(pulse_window=[0.0, 0.05])

    datum = transform(timeseries, prompts)

    assert set(datum.keys()) == {"emg", "targets"}
    assert datum["emg"].shape == (16, 1000)
    assert datum["targets"].shape == (len(GestureType), 1000)
    assert datum["targets"].dtype == torch.float32


def test_discrete_gestures_transform_pulse_region_nonzero() -> None:
    timeseries = _build_timeseries(1000)
    prompts = pd.DataFrame(
        [
            {"name": GestureType.index_press.name, "time": 0.5},
            {"name": GestureType.index_release.name, "time": 0.6},
        ]
    )
    transform = DiscreteGesturesTransform(pulse_window=[0.0, 0.05])

    datum = transform(timeseries, prompts)
    targets = datum["targets"]

    assert targets[GestureType.index_press.value].sum().item() > 0
    assert targets[GestureType.index_release.value].sum().item() > 0


def test_rotation_augmentation_preserves_shape() -> None:
    torch.manual_seed(0)
    augmentation = RotationAugmentation(rotation=2)
    emg = torch.arange(16 * 100, dtype=torch.float32).reshape(16, 100)

    rotated = augmentation(emg)

    assert rotated.shape == emg.shape


def test_rotation_augmentation_identity_when_rotation_zero() -> None:
    augmentation = RotationAugmentation(rotation=0)
    emg = torch.randn(16, 128)

    rotated = augmentation(emg)

    assert torch.equal(rotated, emg)
