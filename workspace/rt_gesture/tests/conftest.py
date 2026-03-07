from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from rt_gesture.checkpoint_utils import load_model_from_lightning_checkpoint
from rt_gesture.event_detector import EventDetector, EventDetectorConfig
from rt_gesture.networks import DiscreteGesturesArchitecture


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MINI_DATA_DIR = Path(
    "/mnt/data/Dataset/generic-neuromotor-interface-data/Discrete Gestures/mini"
)


def _resolve_checkpoint_path() -> Path | None:
    candidates: list[Path] = []
    from_env = os.environ.get("RT_GESTURE_CHECKPOINT_PATH")
    if from_env:
        candidates.append(Path(from_env).expanduser())
    candidates.append(WORKSPACE_ROOT / "checkpoints" / "discrete_gestures" / "model_checkpoint.ckpt")
    candidates.append(
        REPO_ROOT
        / "generic-neuromotor-interface"
        / "emg_models"
        / "discrete_gestures"
        / "model_checkpoint.ckpt"
    )
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


@pytest.fixture
def model() -> DiscreteGesturesArchitecture:
    network = DiscreteGesturesArchitecture()
    network.eval()
    return network


@pytest.fixture
def pretrained_model() -> DiscreteGesturesArchitecture:
    checkpoint_path = _resolve_checkpoint_path()
    if checkpoint_path is None:
        pytest.skip("Checkpoint not found in RT_GESTURE_CHECKPOINT_PATH or workspace/checkpoints/")
    return load_model_from_lightning_checkpoint(checkpoint_path, device="cpu")


@pytest.fixture
def sample_emg() -> torch.Tensor:
    return torch.randn(1, 16, 2000)


@pytest.fixture
def event_detector() -> EventDetector:
    return EventDetector(EventDetectorConfig())


@pytest.fixture
def mini_hdf5_path() -> Path:
    files = sorted(MINI_DATA_DIR.glob("*.hdf5"))
    if not files:
        pytest.skip(f"No mini HDF5 dataset found in {MINI_DATA_DIR}")
    return files[0]
