from __future__ import annotations

from pathlib import Path

import torch

from rt_gesture.checkpoint_utils import _extract_state_dict, load_model_from_lightning_checkpoint
from rt_gesture.networks import DiscreteGesturesArchitecture


def test_extract_state_dict_strips_network_prefix() -> None:
    raw = {
        "state_dict": {
            "network.projection.weight": torch.zeros(9, 512),
            "network.projection.bias": torch.zeros(9),
        }
    }
    cleaned = _extract_state_dict(raw)
    assert "projection.weight" in cleaned
    assert "projection.bias" in cleaned


def test_load_model_from_lightning_checkpoint(tmp_path: Path) -> None:
    model = DiscreteGesturesArchitecture()
    checkpoint = {
        "state_dict": {f"network.{key}": value for key, value in model.state_dict().items()}
    }
    checkpoint_path = tmp_path / "test.ckpt"
    torch.save(checkpoint, checkpoint_path)

    loaded = load_model_from_lightning_checkpoint(checkpoint_path, device="cpu")
    for key, value in model.state_dict().items():
        assert torch.allclose(value, loaded.state_dict()[key])

