from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rt_gesture.config import AppConfig, dict_to_config, load_config, save_config, validate_config
from rt_gesture.constants import DEFAULT_THRESHOLD, ZMQ_RESULT_PORT


def test_load_config_resolves_relative_paths_and_applies_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "default.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = tmp_path / "artifacts" / "model.ckpt"
    data_path = tmp_path / "data" / "sample.hdf5"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    raw = {
        "data_simulator": {"hdf5_path": "../data/sample.hdf5"},
        "inference": {
            "checkpoint_path": "../artifacts/model.ckpt",
            "result_port": ZMQ_RESULT_PORT,
        },
    }
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    loaded = load_config(config_path)

    assert Path(loaded.data_simulator.hdf5_path) == data_path.resolve()
    assert Path(loaded.inference.checkpoint_path) == model_path.resolve()
    assert loaded.inference.threshold == DEFAULT_THRESHOLD


def test_validate_config_rejects_invalid_threshold() -> None:
    config = AppConfig()
    config.inference.threshold = 1.2
    with pytest.raises(ValueError, match="inference.threshold"):
        validate_config(config)


def test_validate_config_rejects_invalid_heartbeat_timeout() -> None:
    config = AppConfig()
    config.gui.heartbeat_timeout_sec = 0.0
    with pytest.raises(ValueError, match="gui.heartbeat_timeout_sec"):
        validate_config(config)


def test_save_and_reload_round_trip(tmp_path: Path) -> None:
    original = AppConfig()
    original.data_simulator.chunk_size = 80
    target_path = tmp_path / "saved_config.yaml"
    save_config(original, target_path)
    loaded = load_config(target_path)
    assert loaded.data_simulator.chunk_size == 80


def test_dict_to_config_handles_empty_input() -> None:
    config = dict_to_config({})
    assert isinstance(config, AppConfig)
    assert config.inference.result_port == ZMQ_RESULT_PORT
