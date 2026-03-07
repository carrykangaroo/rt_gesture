"""YAML-driven training pipeline for discrete gesture model."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from rt_gesture.constants import GestureType
from rt_gesture.data import DataPipelineConfig, make_discrete_gesture_dataloaders
from rt_gesture.networks import DiscreteGesturesArchitecture

log = logging.getLogger(__name__)


class FingerStateMaskGenerator(nn.Module):
    """Generate finger-state masks from press/release pulses."""

    def __init__(self, lpad: int = 0, rpad: int = 7) -> None:
        super().__init__()
        self.lpad = lpad
        self.rpad = rpad

    def forward(self, gesture_labels: torch.Tensor) -> torch.Tensor:
        batch_size, _, time_steps = gesture_labels.shape
        masks = torch.zeros((batch_size, 2, time_steps), device=gesture_labels.device)
        for batch_idx in range(batch_size):
            self._process_finger(
                labels=gesture_labels[batch_idx],
                output=masks[batch_idx, 0],
                press_channel=GestureType.index_press.value,
                release_channel=GestureType.index_release.value,
                time_steps=time_steps,
            )
            self._process_finger(
                labels=gesture_labels[batch_idx],
                output=masks[batch_idx, 1],
                press_channel=GestureType.middle_press.value,
                release_channel=GestureType.middle_release.value,
                time_steps=time_steps,
            )
        return masks

    def _process_finger(
        self,
        labels: torch.Tensor,
        output: torch.Tensor,
        press_channel: int,
        release_channel: int,
        time_steps: int,
    ) -> None:
        zero = torch.zeros(1, device=labels.device)
        press_onsets = torch.nonzero(
            torch.diff(labels[press_channel], prepend=zero) > 0, as_tuple=True
        )[0]
        release_onsets = torch.nonzero(
            torch.diff(labels[release_channel], prepend=zero) > 0, as_tuple=True
        )[0]
        if press_onsets.numel() == 0:
            return
        for press_idx in press_onsets:
            future_releases = release_onsets[release_onsets > press_idx]
            if future_releases.numel() == 0:
                release_idx = time_steps - 1
            else:
                release_idx = int(future_releases[0].item())
            start_idx = max(int(press_idx.item()) - self.lpad, 0)
            end_idx = min(release_idx + self.rpad + 1, time_steps)
            output[start_idx:end_idx] = 1.0


@dataclass
class TrainingConfig:
    data: DataPipelineConfig
    checkpoint_dir: str = "checkpoints"
    run_name: str = "discrete_gestures"
    seed: int = 0
    device: str | None = None
    learning_rate: float = 5e-4
    max_epochs: int = 1
    gradient_clip_val: float = 0.5
    lr_scheduler_milestones: list[int] = field(default_factory=lambda: [25])
    lr_scheduler_factor: float = 0.5
    warmup_start_factor: float = 0.001
    warmup_end_factor: float = 1.0
    warmup_total_epochs: int = 5
    max_train_steps_per_epoch: int | None = None
    max_val_steps_per_epoch: int | None = None


def _dict_to_training_config(raw: dict) -> TrainingConfig:
    data_cfg = DataPipelineConfig(**raw["data"])
    return TrainingConfig(
        data=data_cfg,
        checkpoint_dir=raw.get("checkpoint_dir", "checkpoints"),
        run_name=raw.get("run_name", "discrete_gestures"),
        seed=raw.get("seed", 0),
        device=raw.get("device"),
        learning_rate=raw.get("learning_rate", 5e-4),
        max_epochs=raw.get("max_epochs", 1),
        gradient_clip_val=raw.get("gradient_clip_val", 0.5),
        lr_scheduler_milestones=raw.get("lr_scheduler_milestones", [25]),
        lr_scheduler_factor=raw.get("lr_scheduler_factor", 0.5),
        warmup_start_factor=raw.get("warmup_start_factor", 0.001),
        warmup_end_factor=raw.get("warmup_end_factor", 1.0),
        warmup_total_epochs=raw.get("warmup_total_epochs", 5),
        max_train_steps_per_epoch=raw.get("max_train_steps_per_epoch"),
        max_val_steps_per_epoch=raw.get("max_val_steps_per_epoch"),
    )


def load_training_config(config_path: str | Path) -> TrainingConfig:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _dict_to_training_config(raw)


def save_training_config(config: TrainingConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False, allow_unicode=True)


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def compute_loss(
    model: DiscreteGesturesArchitecture,
    batch: dict[str, torch.Tensor],
    loss_fn: nn.Module,
    mask_generator: FingerStateMaskGenerator,
    device: str,
) -> tuple[torch.Tensor, float, float]:
    emg = batch["emg"].to(device, non_blocking=True)
    targets = batch["targets"].to(device, non_blocking=True)
    aligned_targets = targets[:, :, model.left_context :: model.stride]

    logits = model(emg)
    time_steps = min(logits.shape[2], aligned_targets.shape[2])
    logits = logits[:, :, :time_steps]
    aligned_targets = aligned_targets[:, :, :time_steps]

    release_mask = mask_generator(aligned_targets)
    mask = torch.ones_like(aligned_targets)
    mask[:, [GestureType.index_release.value, GestureType.middle_release.value], :] = (
        release_mask
    )

    raw_loss = loss_fn(logits, aligned_targets)
    loss = (raw_loss * mask).sum() / mask.sum().clamp(min=1.0)

    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        frame_acc = float((preds == aligned_targets).float().mean().item())
        max_probs, pred_class = probs.max(dim=1)
        target_max_probs, target_class = aligned_targets.max(dim=1)
        rest_class = probs.shape[1]
        pred_with_rest = torch.where(
            max_probs >= 0.5,
            pred_class,
            torch.full_like(pred_class, rest_class),
        )
        target_with_rest = torch.where(
            target_max_probs >= 0.5,
            target_class,
            torch.full_like(target_class, rest_class),
        )
        multiclass_acc = float((pred_with_rest == target_with_rest).float().mean().item())
    return loss, frame_acc, multiclass_acc


def _run_epoch(
    model: DiscreteGesturesArchitecture,
    loader: DataLoader,
    device: str,
    loss_fn: nn.Module,
    mask_generator: FingerStateMaskGenerator,
    optimizer: torch.optim.Optimizer | None,
    max_steps: int | None,
    gradient_clip_val: float,
) -> tuple[float, float, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_multiclass_acc = 0.0
    steps = 0
    for step_idx, batch in enumerate(loader):
        if max_steps is not None and step_idx >= max_steps:
            break
        with torch.set_grad_enabled(is_train):
            loss, frame_acc, multiclass_acc = compute_loss(
                model=model,
                batch=batch,
                loss_fn=loss_fn,
                mask_generator=mask_generator,
                device=device,
            )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                optimizer.step()

        total_loss += float(loss.item())
        total_acc += frame_acc
        total_multiclass_acc += multiclass_acc
        steps += 1

    if steps == 0:
        return math.nan, math.nan, math.nan
    return total_loss / steps, total_acc / steps, total_multiclass_acc / steps


def _checkpoint_payload(
    model: DiscreteGesturesArchitecture,
    epoch: int,
    config: TrainingConfig,
    metrics: dict[str, float],
) -> dict:
    return {
        "epoch": epoch,
        "state_dict": {f"network.{k}": v.detach().cpu() for k, v in model.state_dict().items()},
        "metrics": metrics,
        "config": asdict(config),
        "pytorch-lightning_version": "rt_gesture_custom",
    }


def train_discrete_gestures(config: TrainingConfig) -> dict[str, str | float]:
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    train_loader, val_loader, _ = make_discrete_gesture_dataloaders(config.data)

    model = DiscreteGesturesArchitecture().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=config.warmup_start_factor,
        end_factor=config.warmup_end_factor,
        total_iters=max(1, config.warmup_total_epochs),
    )
    multistep = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.lr_scheduler_milestones,
        gamma=config.lr_scheduler_factor,
    )
    scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup, multistep])
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    mask_generator = FingerStateMaskGenerator(lpad=0, rpad=7)

    run_dir = Path(config.checkpoint_dir) / config.run_name / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_training_config(config, run_dir / "training_config.yaml")

    best_val_loss = float("inf")
    best_val_multiclass_acc = float("nan")
    best_checkpoint = run_dir / "best.ckpt"
    last_checkpoint = run_dir / "last.ckpt"

    for epoch in range(1, config.max_epochs + 1):
        train_loss, train_acc, train_multiclass_acc = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            loss_fn=loss_fn,
            mask_generator=mask_generator,
            optimizer=optimizer,
            max_steps=config.max_train_steps_per_epoch,
            gradient_clip_val=config.gradient_clip_val,
        )
        val_loss, val_acc, val_multiclass_acc = _run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            mask_generator=mask_generator,
            optimizer=None,
            max_steps=config.max_val_steps_per_epoch,
            gradient_clip_val=config.gradient_clip_val,
        )
        scheduler.step()
        metrics = {
            "train_loss": train_loss,
            "train_frame_accuracy": train_acc,
            "train_multiclass_accuracy": train_multiclass_acc,
            "val_loss": val_loss,
            "val_frame_accuracy": val_acc,
            "val_multiclass_accuracy": val_multiclass_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        log.info(
            (
                "epoch=%s train_loss=%.6f val_loss=%.6f train_acc=%.4f "
                "val_acc=%.4f train_mc_acc=%.4f val_mc_acc=%.4f lr=%.6g"
            ),
            epoch,
            train_loss,
            val_loss,
            train_acc,
            val_acc,
            train_multiclass_acc,
            val_multiclass_acc,
            metrics["lr"],
        )
        torch.save(_checkpoint_payload(model, epoch, config, metrics), last_checkpoint)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_multiclass_acc = val_multiclass_acc
            torch.save(_checkpoint_payload(model, epoch, config, metrics), best_checkpoint)

    summary = {
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
        "best_val_loss": float(best_val_loss),
        "best_val_multiclass_accuracy": float(best_val_multiclass_acc),
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/training.yaml", help="Training YAML config path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = load_training_config(args.config)
    summary = train_discrete_gestures(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
