from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import zmq

from rt_gesture.config import DataSimulatorConfig, InferenceConfig
from rt_gesture.constants import MsgType
from rt_gesture.data_simulator import DataSimulator
from rt_gesture.inference_engine import InferenceEngine
from rt_gesture.zmq_transport import ZmqSubscriber


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_hdf5(path: Path, num_samples: int = 400) -> None:
    dtype = np.dtype([("emg", np.float32, (16,)), ("time", np.float64)])
    data = np.zeros(num_samples, dtype=dtype)
    data["emg"] = np.random.randn(num_samples, 16).astype(np.float32)
    data["time"] = np.arange(num_samples, dtype=np.float64) / 2000.0 + 10.0
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=data)


def test_end_to_end_simulator_to_inference_pipeline(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "stream.hdf5"
    _build_hdf5(hdf5_path, num_samples=320)

    emg_port = _pick_free_port()
    gt_port = _pick_free_port()
    result_port = _pick_free_port()
    control_port = _pick_free_port()

    sim_config = DataSimulatorConfig(
        hdf5_path=str(hdf5_path),
        chunk_size=40,
        emg_port=emg_port,
        gt_port=gt_port,
        control_port=control_port,
    )
    inf_config = InferenceConfig(
        checkpoint_path="",
        device="cpu",
        emg_port=emg_port,
        result_port=result_port,
        control_port=control_port,
        threshold=0.2,
        rejection_threshold=0.2,
        rest_threshold=0.05,
        debounce_sec=0.01,
        data_timeout_ms=200,
        warm_up_frames=0,
        max_consecutive_drops=3,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    simulator = DataSimulator(sim_config)
    simulator.load_data(hdf5_path)
    engine = InferenceEngine(inf_config, run_dir=run_dir)

    ctx = zmq.Context()
    result_sub = ZmqSubscriber(ctx, port=result_port, connect=True)
    time.sleep(0.1)

    inference_thread = threading.Thread(target=engine.run, kwargs={"max_messages": 6}, daemon=True)
    inference_thread.start()
    time.sleep(0.1)

    simulator_thread = threading.Thread(target=simulator.run, kwargs={"max_chunks": 6}, daemon=True)
    simulator_thread.start()

    probability_messages = 0
    event_messages = 0
    deadline = time.time() + 6.0
    while time.time() < deadline and (probability_messages < 2 or event_messages < 1):
        received = result_sub.recv(timeout_ms=200)
        if received is None:
            continue
        header, array = received
        msg_type = header.get("msg_type")
        if msg_type == MsgType.PROBABILITIES:
            probability_messages += 1
            assert array is not None
            assert array.shape[0] == 9
            for key in ("transport_ms", "infer_ms", "pipeline_ms"):
                assert key in header
                assert float(header[key]) >= 0.0
        if msg_type == MsgType.GESTURE_EVENT:
            event_messages += 1
            assert "gesture" in header
            assert "confidence" in header
            for key in ("transport_ms", "infer_ms", "post_ms", "pipeline_ms"):
                assert key in header
                assert float(header[key]) >= 0.0

    simulator_thread.join(timeout=3)
    inference_thread.join(timeout=3)

    result_sub.close()
    ctx.term()
    simulator.cleanup()
    engine.cleanup()

    assert probability_messages >= 1
    assert event_messages >= 1
    assert (run_dir / "predictions.npz").exists()
    assert (run_dir / "events.jsonl").exists()
