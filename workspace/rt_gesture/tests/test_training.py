from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from rt_gesture.data import DataPipelineConfig
from rt_gesture.networks import DiscreteGesturesArchitecture
from rt_gesture.train import (
    FingerStateMaskGenerator,
    TrainingConfig,
    compute_loss,
    load_training_config,
    train_discrete_gestures,
)


def test_finger_state_mask_generator_shapes() -> None:
    generator = FingerStateMaskGenerator(lpad=0, rpad=1)
    labels = torch.zeros(2, 9, 12)
    labels[0, 0, 3:5] = 1.0
    labels[0, 1, 8:10] = 1.0
    masks = generator(labels)
    assert masks.shape == (2, 2, 12)
    assert masks[0, 0].sum().item() > 0


def test_compute_loss_runs_on_synthetic_batch() -> None:
    model = DiscreteGesturesArchitecture(
        conv_output_channels=64,
        lstm_hidden_size=32,
        lstm_num_layers=2,
    )
    batch = {
        "emg": torch.randn(2, 16, 160),
        "targets": torch.zeros(2, 9, 160),
    }
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    mask_generator = FingerStateMaskGenerator()
    loss, frame_acc, multiclass_acc = compute_loss(
        model=model,
        batch=batch,
        loss_fn=loss_fn,
        mask_generator=mask_generator,
        device="cpu",
    )
    assert float(loss.item()) >= 0.0
    assert 0.0 <= frame_acc <= 1.0
    assert 0.0 <= multiclass_acc <= 1.0


def test_training_summary_contains_epoch_history(monkeypatch, tmp_path: Path) -> None:
    class _DummyDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {
                "emg": torch.randn(16, 160),
                "targets": torch.zeros(9, 160),
            }

    def _fake_dataloaders(_config: DataPipelineConfig):
        loader = torch.utils.data.DataLoader(_DummyDataset(), batch_size=1, shuffle=False)
        return loader, loader, None

    monkeypatch.setattr("rt_gesture.train.make_discrete_gesture_dataloaders", _fake_dataloaders)

    config = TrainingConfig(
        data=DataPipelineConfig(data_location="unused", split_csv="unused"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        run_name="unit_test_run",
        device="cpu",
        max_epochs=2,
        max_train_steps_per_epoch=1,
        max_val_steps_per_epoch=1,
        warmup_total_epochs=1,
    )

    summary = train_discrete_gestures(config)
    assert "epoch_history" in summary
    epoch_history = summary["epoch_history"]
    assert isinstance(epoch_history, list)
    assert len(epoch_history) == 2
    assert [entry["epoch"] for entry in epoch_history] == [1, 2]

    summary_path = Path(summary["run_dir"]) / "training_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "epoch_history" in payload
    assert len(payload["epoch_history"]) == 2


def test_load_training_config_expands_data_paths_without_rebasing_checkpoint_dir(
    tmp_path: Path,
) -> None:
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    config_path = cfg_dir / "training.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "data_location": "${RT_GESTURE_DATA_ROOT}/mini",
                    "split_csv": "../split.csv",
                },
                "checkpoint_dir": "checkpoints",
            }
        ),
        encoding="utf-8",
    )

    config = load_training_config(config_path)
    assert config.checkpoint_dir == "checkpoints"
    assert config.data.split_csv == str((cfg_dir / "../split.csv").resolve())
