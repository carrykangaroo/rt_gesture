from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from rt_gesture.cler import compute_cler
from rt_gesture.constants import GestureType
from rt_gesture.evaluate import load_evaluation_config, run_full_forward, run_streaming_forward
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


def test_full_forward_is_stable_across_chunk_sizes() -> None:
    model = DiscreteGesturesArchitecture(
        conv_output_channels=64,
        lstm_hidden_size=32,
        lstm_num_layers=2,
    )
    model.eval()
    emg = torch.randn(1, 16, 420)
    times = np.arange(420, dtype=np.float64) / 2000.0

    probs_small, times_small = run_full_forward(model, emg, times, device="cpu", chunk_size=35)
    probs_large, times_large = run_full_forward(model, emg, times, device="cpu", chunk_size=200)

    np.testing.assert_allclose(probs_small, probs_large, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(times_small, times_large, rtol=0.0, atol=0.0)


def test_load_evaluation_config_resolves_relative_paths(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = tmp_path / "checkpoints" / "model.ckpt"
    hdf5_path = tmp_path / "data" / "sample.hdf5"
    report_path = tmp_path / "logs" / "evaluation.json"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)

    raw = {
        "checkpoint_path": "../checkpoints/model.ckpt",
        "hdf5_path": "../data/sample.hdf5",
        "device": "cpu",
        "chunk_size": 40,
        "report_path": "../logs/evaluation.json",
    }
    config_file = cfg_dir / "evaluation.yaml"
    config_file.write_text(yaml.safe_dump(raw), encoding="utf-8")

    loaded = load_evaluation_config(config_file)

    assert Path(loaded.checkpoint_path) == ckpt_path.resolve()
    assert Path(loaded.hdf5_path) == hdf5_path.resolve()
    assert Path(loaded.report_path) == report_path.resolve()


@pytest.mark.slow
def test_cler_baseline_alignment_on_mini_window(
    pretrained_model: DiscreteGesturesArchitecture,
    mini_hdf5_path: Path,
) -> None:
    """
    Validate full-forward vs streaming CLER alignment on a real-data window.

    This test is marked slow and skipped unless explicitly enabled:
    RT_GESTURE_RUN_SLOW=1 pytest -m slow
    """
    if os.environ.get("RT_GESTURE_RUN_SLOW") != "1":
        pytest.skip("Set RT_GESTURE_RUN_SLOW=1 to run slow CLER alignment test")

    prompts = pd.read_hdf(mini_hdf5_path, "prompts")
    if prompts.empty:
        pytest.skip("No prompts available in mini dataset")
    prompts = prompts[prompts["name"].isin([g.name for g in GestureType])]
    if prompts.empty:
        pytest.skip("No discrete-gesture prompts available")

    probe = prompts.head(20)
    t_start = float(probe["time"].min() - 1.0)
    t_end = float(probe["time"].max() + 1.0)

    with h5py.File(mini_hdf5_path, "r") as handle:
        dataset = handle["data"]
        all_times = dataset["time"]
        start_idx, end_idx = np.searchsorted(all_times, [t_start, t_end])
        segment = dataset[start_idx:end_idx]

    if len(segment) < 100:
        pytest.skip("Selected real-data segment is too short for CLER alignment")

    emg = torch.from_numpy(np.stack(segment["emg"], axis=0).T).float().unsqueeze(0)
    times = segment["time"].astype(np.float64)
    prompts_window = prompts[prompts["time"].between(float(times[0]), float(times[-1]))]
    if prompts_window.empty:
        pytest.skip("No prompts fall into selected real-data segment")

    model = pretrained_model.eval()
    full_probs, full_times = run_full_forward(model, emg, times, device="cpu")
    stream_probs, stream_times = run_streaming_forward(
        model=model,
        emg=emg,
        times=times,
        chunk_size=40,
        device="cpu",
    )

    full_cler = compute_cler(full_probs, full_times, prompts_window)
    stream_cler = compute_cler(stream_probs, stream_times, prompts_window)

    assert abs(full_cler - stream_cler) < 0.01
