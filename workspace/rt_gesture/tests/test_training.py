from __future__ import annotations

import torch

from rt_gesture.networks import DiscreteGesturesArchitecture
from rt_gesture.train import FingerStateMaskGenerator, compute_loss


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
