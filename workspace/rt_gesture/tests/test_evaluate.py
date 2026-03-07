from __future__ import annotations

import numpy as np
import torch

from rt_gesture.evaluate import run_full_forward, run_streaming_forward
from rt_gesture.networks import DiscreteGesturesArchitecture


def test_streaming_and_full_forward_shapes_are_consistent() -> None:
    model = DiscreteGesturesArchitecture(
        conv_output_channels=64,
        lstm_hidden_size=32,
        lstm_num_layers=2,
    )
    model.eval()
    emg = torch.randn(1, 16, 260)
    times = np.arange(260, dtype=np.float64) / 2000.0

    full_probs, full_times = run_full_forward(model, emg, times, device="cpu")
    stream_probs, stream_times = run_streaming_forward(
        model=model,
        emg=emg,
        times=times,
        chunk_size=40,
        device="cpu",
    )
    assert full_probs.shape == stream_probs.shape
    assert full_times.shape == stream_times.shape

