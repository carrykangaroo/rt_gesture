#!/usr/bin/env python3
"""Run a timed realtime session and emit stability/latency report."""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import zmq

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rt_gesture.config import AppConfig, load_config
from rt_gesture.constants import EMG_SAMPLE_RATE, MsgType
from rt_gesture.logger import setup_logging
from rt_gesture.main import _send_shutdown, run_data_simulator, run_inference_engine
from rt_gesture.zmq_transport import ZmqSubscriber

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="YAML config path.")
    parser.add_argument("--duration-sec", type=float, default=180.0, help="Target run duration.")
    parser.add_argument(
        "--sample-interval-sec",
        type=float,
        default=1.0,
        help="Sampling interval for memory/latency polling.",
    )
    parser.add_argument(
        "--report-path",
        default="",
        help="Optional report output path. Defaults to <run_dir>/stability_report.json",
    )
    return parser.parse_args()


def _required_chunks(config: AppConfig, duration_sec: float) -> int:
    chunks = duration_sec * EMG_SAMPLE_RATE / config.data_simulator.chunk_size
    return int(math.ceil(chunks)) + 10


def _read_rss_mb(pid: int) -> float | None:
    status_path = Path("/proc") / str(pid) / "status"
    if not status_path.exists():
        return None
    for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                return float(parts[1]) / 1024.0
    return None


def _series_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _start_processes(config: AppConfig, run_dir: Path) -> tuple[mp.Process, mp.Process]:
    inference_proc = mp.Process(
        target=run_inference_engine,
        args=(config, str(run_dir)),
        name="InferenceEngine",
        daemon=False,
    )
    data_proc = mp.Process(
        target=run_data_simulator,
        args=(config,),
        name="DataSimulator",
        daemon=False,
    )
    inference_proc.start()
    time.sleep(0.5)
    data_proc.start()
    return inference_proc, data_proc


def run_validation(config_path: str, duration_sec: float, sample_interval_sec: float) -> dict[str, object]:
    config = load_config(config_path)
    required_chunks = _required_chunks(config, duration_sec)
    config.data_simulator.max_chunks = required_chunks
    config.inference.max_messages = required_chunks + 50

    run_dir = setup_logging(config.logging.log_dir, config.logging.log_level)
    log.info(
        "Stability validation start config=%s duration_sec=%.1f required_chunks=%d",
        config_path,
        duration_sec,
        required_chunks,
    )

    inference_proc, data_proc = _start_processes(config, run_dir)

    ctx = zmq.Context()
    result_sub = ZmqSubscriber(ctx, config.inference.result_port)
    latencies: dict[str, list[float]] = {
        "transport_ms": [],
        "infer_ms": [],
        "post_ms": [],
        "pipeline_ms": [],
    }
    counts = {
        "probabilities": 0,
        "gesture_event": 0,
        "heartbeat": 0,
    }
    rss_samples: list[tuple[float, float]] = []

    start = time.monotonic()
    try:
        while True:
            elapsed = time.monotonic() - start
            if elapsed >= duration_sec:
                break
            if not inference_proc.is_alive() and not data_proc.is_alive():
                break

            rss_mb = _read_rss_mb(inference_proc.pid)
            if rss_mb is not None:
                rss_samples.append((elapsed, rss_mb))

            next_tick = time.monotonic() + sample_interval_sec
            while time.monotonic() < next_tick:
                received = result_sub.recv(timeout_ms=20)
                if received is None:
                    continue
                header, _ = received
                msg_type = header.get("msg_type")
                if msg_type == MsgType.PROBABILITIES:
                    counts["probabilities"] += 1
                elif msg_type == MsgType.GESTURE_EVENT:
                    counts["gesture_event"] += 1
                elif msg_type == MsgType.HEARTBEAT:
                    counts["heartbeat"] += 1
                for key in latencies:
                    if key in header:
                        latencies[key].append(float(header[key]))
    finally:
        _send_shutdown(config.inference.control_port)
        data_proc.join(timeout=5.0)
        inference_proc.join(timeout=config.shutdown_timeout_sec)
        for proc in (data_proc, inference_proc):
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
        result_sub.close()
        ctx.term()

    observed_duration_sec = float(time.monotonic() - start)
    rss_values = [sample[1] for sample in rss_samples]
    memory_stats = {
        "samples": len(rss_samples),
        "start_mb": float(rss_values[0]) if rss_values else None,
        "end_mb": float(rss_values[-1]) if rss_values else None,
        "peak_mb": float(max(rss_values)) if rss_values else None,
        "delta_mb": float(rss_values[-1] - rss_values[0]) if len(rss_values) >= 2 else None,
    }

    return {
        "config_path": str(Path(config_path).resolve()),
        "run_dir": str(run_dir.resolve()),
        "duration_requested_sec": float(duration_sec),
        "duration_observed_sec": observed_duration_sec,
        "sample_interval_sec": float(sample_interval_sec),
        "required_chunks": int(required_chunks),
        "chunk_size": int(config.data_simulator.chunk_size),
        "counts": counts,
        "latency_stats": {name: _series_stats(values) for name, values in latencies.items()},
        "memory_stats": memory_stats,
        "inference_exit_code": inference_proc.exitcode,
        "simulator_exit_code": data_proc.exitcode,
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    report = run_validation(
        config_path=args.config,
        duration_sec=args.duration_sec,
        sample_interval_sec=args.sample_interval_sec,
    )

    report_path = Path(args.report_path) if args.report_path else Path(report["run_dir"]) / "stability_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    log.info("Stability report written to %s", report_path)
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
