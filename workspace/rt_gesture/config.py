"""Config management for RT-Gesture."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rt_gesture.constants import (
    DATA_TIMEOUT_MS,
    DEFAULT_HEARTBEAT_INTERVAL_SEC,
    DEFAULT_HEARTBEAT_TIMEOUT_SEC,
    DEFAULT_DEBOUNCE_SEC,
    DEFAULT_REJECTION_THRESHOLD,
    DEFAULT_REST_THRESHOLD,
    DEFAULT_THRESHOLD,
    LATENCY_WARN_INFER_MS,
    LATENCY_WARN_PIPELINE_MS,
    LATENCY_WARN_POST_MS,
    LATENCY_WARN_TRANSPORT_MS,
    MAX_CONSECUTIVE_DROPS,
    SHUTDOWN_TIMEOUT_SEC,
    WARM_UP_FRAMES,
    ZMQ_CONTROL_PORT,
    ZMQ_EMG_PORT,
    ZMQ_GT_PORT,
    ZMQ_RESULT_PORT,
)


@dataclass
class DataSimulatorConfig:
    hdf5_path: str = ""
    chunk_size: int = 40
    max_chunks: int | None = None
    emg_port: int = ZMQ_EMG_PORT
    gt_port: int = ZMQ_GT_PORT
    control_port: int = ZMQ_CONTROL_PORT


@dataclass
class InferenceConfig:
    checkpoint_path: str = ""
    device: str | None = None
    emg_port: int = ZMQ_EMG_PORT
    result_port: int = ZMQ_RESULT_PORT
    control_port: int = ZMQ_CONTROL_PORT
    threshold: float = DEFAULT_THRESHOLD
    rejection_threshold: float = DEFAULT_REJECTION_THRESHOLD
    rest_threshold: float = DEFAULT_REST_THRESHOLD
    debounce_sec: float = DEFAULT_DEBOUNCE_SEC
    data_timeout_ms: int = DATA_TIMEOUT_MS
    warm_up_frames: int = WARM_UP_FRAMES
    max_consecutive_drops: int = MAX_CONSECUTIVE_DROPS
    max_messages: int | None = None
    heartbeat_interval_sec: float = DEFAULT_HEARTBEAT_INTERVAL_SEC
    latency_warn_transport_ms: float = LATENCY_WARN_TRANSPORT_MS
    latency_warn_infer_ms: float = LATENCY_WARN_INFER_MS
    latency_warn_post_ms: float = LATENCY_WARN_POST_MS
    latency_warn_pipeline_ms: float = LATENCY_WARN_PIPELINE_MS


@dataclass
class GUIConfig:
    emg_port: int = ZMQ_EMG_PORT
    result_port: int = ZMQ_RESULT_PORT
    gt_port: int = ZMQ_GT_PORT
    control_port: int = ZMQ_CONTROL_PORT
    plot_duration_sec: float = 5.0
    refresh_rate_hz: int = 30
    heartbeat_timeout_sec: float = DEFAULT_HEARTBEAT_TIMEOUT_SEC


@dataclass
class LogConfig:
    log_dir: str = "logs"
    log_level: str = "INFO"
    save_predictions: bool = True
    save_events: bool = True


@dataclass
class AppConfig:
    data_simulator: DataSimulatorConfig = field(default_factory=DataSimulatorConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    logging: LogConfig = field(default_factory=LogConfig)
    shutdown_timeout_sec: float = SHUTDOWN_TIMEOUT_SEC


def resolve_path(path_str: str, base_dir: Path | None = None) -> str:
    """Resolve an optional path string into absolute form."""
    if not path_str:
        return path_str
    path = Path(path_str).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return str(path.resolve())


def dict_to_config(raw: dict[str, Any], base_dir: Path | None = None) -> AppConfig:
    """Map a raw nested dictionary to AppConfig."""
    config = AppConfig(
        data_simulator=DataSimulatorConfig(**raw.get("data_simulator", {})),
        inference=InferenceConfig(**raw.get("inference", {})),
        gui=GUIConfig(**raw.get("gui", {})),
        logging=LogConfig(**raw.get("logging", {})),
        shutdown_timeout_sec=raw.get("shutdown_timeout_sec", SHUTDOWN_TIMEOUT_SEC),
    )

    config.data_simulator.hdf5_path = resolve_path(
        config.data_simulator.hdf5_path,
        base_dir=base_dir,
    )
    config.inference.checkpoint_path = resolve_path(
        config.inference.checkpoint_path,
        base_dir=base_dir,
    )
    return config


def validate_config(config: AppConfig) -> None:
    """Raise ValueError when config has invalid values."""
    if config.data_simulator.chunk_size <= 0:
        raise ValueError("data_simulator.chunk_size must be > 0")

    for field_name, value in (
        ("inference.threshold", config.inference.threshold),
        ("inference.rejection_threshold", config.inference.rejection_threshold),
        ("inference.rest_threshold", config.inference.rest_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field_name} must be in [0, 1], got {value}")

    if config.inference.debounce_sec < 0:
        raise ValueError("inference.debounce_sec must be >= 0")
    if config.inference.data_timeout_ms <= 0:
        raise ValueError("inference.data_timeout_ms must be > 0")
    if config.inference.warm_up_frames < 0:
        raise ValueError("inference.warm_up_frames must be >= 0")
    if config.inference.heartbeat_interval_sec <= 0:
        raise ValueError("inference.heartbeat_interval_sec must be > 0")
    if config.gui.heartbeat_timeout_sec <= 0:
        raise ValueError("gui.heartbeat_timeout_sec must be > 0")
    if config.data_simulator.max_chunks is not None and config.data_simulator.max_chunks <= 0:
        raise ValueError("data_simulator.max_chunks must be > 0 when set")
    if config.inference.max_messages is not None and config.inference.max_messages <= 0:
        raise ValueError("inference.max_messages must be > 0 when set")
    for field_name, value in (
        ("inference.latency_warn_transport_ms", config.inference.latency_warn_transport_ms),
        ("inference.latency_warn_infer_ms", config.inference.latency_warn_infer_ms),
        ("inference.latency_warn_post_ms", config.inference.latency_warn_post_ms),
        ("inference.latency_warn_pipeline_ms", config.inference.latency_warn_pipeline_ms),
    ):
        if value <= 0:
            raise ValueError(f"{field_name} must be > 0")
    if config.shutdown_timeout_sec <= 0:
        raise ValueError("shutdown_timeout_sec must be > 0")

    port_values = [
        config.data_simulator.emg_port,
        config.data_simulator.gt_port,
        config.data_simulator.control_port,
        config.inference.emg_port,
        config.inference.result_port,
        config.inference.control_port,
        config.gui.emg_port,
        config.gui.result_port,
        config.gui.gt_port,
        config.gui.control_port,
    ]
    for port in port_values:
        if not 1 <= port <= 65535:
            raise ValueError(f"port out of range: {port}")


def load_config(config_path: str | Path) -> AppConfig:
    """Load a YAML config and return a validated AppConfig."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw_data = yaml.safe_load(handle) or {}
    config = dict_to_config(raw_data, base_dir=path.parent)
    validate_config(config)
    return config


def save_config(config: AppConfig, path: str | Path) -> None:
    """Serialize AppConfig to YAML."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            asdict(config),
            handle,
            allow_unicode=True,
            sort_keys=False,
        )
