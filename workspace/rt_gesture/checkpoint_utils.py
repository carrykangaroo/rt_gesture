"""Checkpoint loading utilities for DiscreteGesturesArchitecture."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import Any

import torch

import rt_gesture.networks as rt_networks
from rt_gesture.networks import DiscreteGesturesArchitecture

log = logging.getLogger(__name__)


def auto_select_device() -> str:
    """Return CUDA device when available, else CPU."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        log.info("CUDA available: %s", device_name)
        return "cuda"
    log.info("CUDA unavailable, using CPU")
    return "cpu"


def _register_legacy_network_module_alias() -> None:
    """Expose legacy module path used by historical Lightning checkpoints."""
    legacy_pkg_name = "generic_neuromotor_interface"
    legacy_mod_name = f"{legacy_pkg_name}.networks"
    if legacy_mod_name in sys.modules:
        return

    legacy_pkg = sys.modules.get(legacy_pkg_name)
    if legacy_pkg is None:
        legacy_pkg = types.ModuleType(legacy_pkg_name)
        sys.modules[legacy_pkg_name] = legacy_pkg

    legacy_networks = types.ModuleType(legacy_mod_name)
    for name in ("DiscreteGesturesArchitecture", "ReinhardCompression", "WristArchitecture"):
        if hasattr(rt_networks, name):
            setattr(legacy_networks, name, getattr(rt_networks, name))
    legacy_pkg.networks = legacy_networks
    sys.modules[legacy_mod_name] = legacy_networks


def _extract_state_dict(raw_checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "state_dict" in raw_checkpoint:
        source = raw_checkpoint["state_dict"]
    else:
        source = raw_checkpoint

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in source.items():
        normalized_key = key
        if normalized_key.startswith("network."):
            normalized_key = normalized_key[len("network.") :]
        cleaned[normalized_key] = value
    return cleaned


def load_model_from_lightning_checkpoint(
    checkpoint_path: str | Path,
    device: str | None = None,
) -> DiscreteGesturesArchitecture:
    """Load model weights from a Lightning checkpoint."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    resolved_device = device or auto_select_device()
    _register_legacy_network_module_alias()
    checkpoint = torch.load(path, map_location=resolved_device, weights_only=False)
    state_dict = _extract_state_dict(checkpoint)

    model = DiscreteGesturesArchitecture()
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    if missing_keys:
        raise RuntimeError(f"missing model keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected_keys}")

    model.eval()
    model.to(resolved_device)
    log.info(
        "Loaded checkpoint %s with %d parameters on %s",
        path,
        sum(parameter.numel() for parameter in model.parameters()),
        resolved_device,
    )
    return model
