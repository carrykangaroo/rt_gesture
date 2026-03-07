from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import pytest
import zmq

from rt_gesture.config import DataSimulatorConfig
from rt_gesture.constants import MsgType
from rt_gesture.data_simulator import DataSimulator
from rt_gesture.zmq_transport import ZmqPublisher, ZmqSubscriber


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_minimal_hdf5(path: Path, num_samples: int = 120) -> None:
    dtype = np.dtype([("emg", np.float32, (16,)), ("time", np.float64)])
    data = np.zeros(num_samples, dtype=dtype)
    data["emg"] = np.random.randn(num_samples, 16).astype(np.float32)
    data["time"] = np.arange(num_samples, dtype=np.float64) / 2000.0 + 1000.0
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=data)


def test_data_simulator_replays_chunks(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "mini.hdf5"
    _build_minimal_hdf5(hdf5_path)

    emg_port = _pick_free_port()
    gt_port = _pick_free_port()
    control_port = _pick_free_port()
    config = DataSimulatorConfig(
        hdf5_path=str(hdf5_path),
        chunk_size=20,
        emg_port=emg_port,
        gt_port=gt_port,
        control_port=control_port,
    )

    simulator = DataSimulator(config)
    simulator.load_data(hdf5_path)

    ctx = zmq.Context()
    sub = ZmqSubscriber(ctx, port=emg_port, connect=True)
    time.sleep(0.05)

    thread = threading.Thread(target=simulator.run, kwargs={"max_chunks": 2}, daemon=True)
    thread.start()

    first = sub.recv(timeout_ms=1000)
    second = sub.recv(timeout_ms=1000)
    assert first is not None
    assert second is not None
    assert first[1] is not None and first[1].shape == (16, 20)
    assert second[1] is not None and second[1].shape == (16, 20)
    assert first[0]["sample_offset"] == 0
    assert second[0]["sample_offset"] == 20

    thread.join(timeout=2)
    sub.close()
    ctx.term()


@pytest.mark.integration
def test_data_simulator_stops_on_shutdown_control(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "mini_shutdown.hdf5"
    _build_minimal_hdf5(hdf5_path, num_samples=2000)

    emg_port = _pick_free_port()
    gt_port = _pick_free_port()
    control_port = _pick_free_port()
    config = DataSimulatorConfig(
        hdf5_path=str(hdf5_path),
        chunk_size=20,
        emg_port=emg_port,
        gt_port=gt_port,
        control_port=control_port,
    )

    simulator = DataSimulator(config)
    simulator.load_data(hdf5_path)

    thread = threading.Thread(target=simulator.run, daemon=True)
    thread.start()
    time.sleep(0.1)

    ctx = zmq.Context()
    control_pub = ZmqPublisher(ctx, port=control_port, bind=True)
    time.sleep(0.1)
    control_pub.send(MsgType.SHUTDOWN)

    thread.join(timeout=3)
    control_pub.close()
    ctx.term()
    simulator.cleanup()

    assert not thread.is_alive()
