from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from rt_gesture.logger import EventLogger, PredictionStore, setup_logging


def test_setup_logging_creates_runtime_log(tmp_path: Path) -> None:
    run_dir = setup_logging(tmp_path / "logs", log_level="INFO")
    logger = logging.getLogger("rt_gesture.tests")
    logger.info("logging smoke test")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert (run_dir / "runtime.log").exists()


def test_event_logger_writes_jsonl(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    event_logger = EventLogger(run_dir)
    event_logger.log_event(
        "index_press",
        timestamp=0.125,
        confidence=0.88,
        transport_ms=1.234,
        infer_ms=2.345,
        post_ms=0.456,
        pipeline_ms=4.035,
    )
    event_logger.close()

    lines = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["gesture"] == "index_press"
    assert payload["timestamp"] == 0.125
    assert payload["transport_ms"] == 1.234
    assert payload["infer_ms"] == 2.345
    assert payload["post_ms"] == 0.456
    assert payload["pipeline_ms"] == 4.035


def test_prediction_store_saves_npz(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    probs = np.random.rand(9, 4).astype(np.float32)
    times = np.array([0.0, 0.005, 0.01, 0.015], dtype=np.float64)
    store.append(probs=probs, times=times)
    path = store.save()
    assert path is not None
    with np.load(path) as data:
        assert data["probs"].shape == (9, 4)
        assert data["times"].shape == (4,)
