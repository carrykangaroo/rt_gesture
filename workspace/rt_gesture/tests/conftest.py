from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rt_gesture.checkpoint_utils import load_model_from_lightning_checkpoint
from rt_gesture.event_detector import EventDetector, EventDetectorConfig
from rt_gesture.networks import DiscreteGesturesArchitecture


REPO_ROOT = Path(__file__).resolve().parents[3]
MINI_DATA_DIR = Path(
    "/mnt/data/Dataset/generic-neuromotor-interface-data/Discrete Gestures/mini"
)
CHECKPOINT_PATH = (
    REPO_ROOT
    / "generic-neuromotor-interface"
    / "emg_models"
    / "discrete_gestures"
    / "model_checkpoint.ckpt"
)


@pytest.fixture
def model() -> DiscreteGesturesArchitecture:
    network = DiscreteGesturesArchitecture()
    network.eval()
    return network


@pytest.fixture
def pretrained_model() -> DiscreteGesturesArchitecture:
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found: {CHECKPOINT_PATH}")
    return load_model_from_lightning_checkpoint(CHECKPOINT_PATH, device="cpu")


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

