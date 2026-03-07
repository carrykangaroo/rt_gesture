"""Evaluation helpers for CLER comparison and checkpoint validation."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from rt_gesture.checkpoint_utils import load_model_from_lightning_checkpoint
from rt_gesture.cler import DEBOUNCE, THRESHOLD, compute_cler
from rt_gesture.data import load_full_discrete_gesture_recording
from rt_gesture.networks import DiscreteGesturesArchitecture

log = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    checkpoint_path: str
    hdf5_path: str
    device: str = "cpu"
    chunk_size: int = 40
    threshold: float = THRESHOLD
    debounce: float = DEBOUNCE
    report_path: str = "logs/evaluation_report.json"


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return EvaluationConfig(**raw)


def run_full_forward(
    model: DiscreteGesturesArchitecture,
    emg: torch.Tensor,
    times: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        logits = model(emg.to(device))
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    aligned_times = times[model.left_context :: model.stride][: probs.shape[1]]
    return probs.astype(np.float32), aligned_times.astype(np.float64)


def run_streaming_forward(
    model: DiscreteGesturesArchitecture,
    emg: torch.Tensor,
    times: np.ndarray,
    chunk_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    conv_history = torch.zeros(1, emg.shape[1], 0, device=device, dtype=torch.float32)
    lstm_state = None
    outputs: list[np.ndarray] = []
    for start in range(0, emg.shape[2], chunk_size):
        chunk = emg[:, :, start : start + chunk_size].to(device)
        with torch.no_grad():
            logits, conv_history, lstm_state = model.forward_streaming(
                new_samples=chunk,
                conv_history=conv_history,
                lstm_state=lstm_state,
            )
        if logits.shape[2] > 0:
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            outputs.append(probs.astype(np.float32))
    if outputs:
        concatenated = np.concatenate(outputs, axis=1)
    else:
        concatenated = np.zeros((9, 0), dtype=np.float32)
    aligned_times = times[model.left_context :: model.stride][: concatenated.shape[1]]
    return concatenated, aligned_times.astype(np.float64)


def evaluate_cler_consistency(config: EvaluationConfig) -> dict[str, float | str]:
    model = load_model_from_lightning_checkpoint(config.checkpoint_path, device=config.device)
    model.eval()

    emg, times, prompts = load_full_discrete_gesture_recording(config.hdf5_path)
    start = time.perf_counter()
    full_probs, full_times = run_full_forward(model, emg, times, config.device)
    full_elapsed_ms = (time.perf_counter() - start) * 1000.0
    full_cler = float(compute_cler(full_probs, full_times, prompts))

    start = time.perf_counter()
    stream_probs, stream_times = run_streaming_forward(
        model=model,
        emg=emg,
        times=times,
        chunk_size=config.chunk_size,
        device=config.device,
    )
    stream_elapsed_ms = (time.perf_counter() - start) * 1000.0
    stream_cler = float(compute_cler(stream_probs, stream_times, prompts))

    result = {
        "checkpoint_path": str(config.checkpoint_path),
        "hdf5_path": str(config.hdf5_path),
        "full_cler": full_cler,
        "streaming_cler": stream_cler,
        "cler_abs_diff": abs(full_cler - stream_cler),
        "full_forward_ms": full_elapsed_ms,
        "streaming_forward_ms": stream_elapsed_ms,
        "frames_full": float(full_probs.shape[1]),
        "frames_streaming": float(stream_probs.shape[1]),
    }
    report_path = Path(config.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/evaluation.yaml", help="Evaluation YAML config path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = load_evaluation_config(args.config)
    result = evaluate_cler_consistency(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
