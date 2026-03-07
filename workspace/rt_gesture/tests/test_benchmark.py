from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest
import torch
import zmq

from rt_gesture.config import InferenceConfig
from rt_gesture.constants import MsgType
from rt_gesture.event_detector import EventDetector
from rt_gesture.inference_engine import InferenceEngine
from rt_gesture.networks import DiscreteGesturesArchitecture
from rt_gesture.zmq_transport import ZmqPublisher, ZmqSubscriber


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.benchmark
def test_forward_streaming_latency_budget() -> None:
    model = DiscreteGesturesArchitecture()
    model.eval()
    history = torch.zeros(1, 16, 0)
    state = None
    chunk = torch.randn(1, 16, 40)

    with torch.no_grad():
        for _ in range(20):
            _, history, state = model.forward_streaming(chunk, history, state)

        iterations = 200
        samples_ms: list[float] = []
        for _ in range(iterations):
            step_start = time.perf_counter()
            _, history, state = model.forward_streaming(chunk, history, state)
            samples_ms.append((time.perf_counter() - step_start) * 1000.0)

    assert float(np.mean(samples_ms)) < 15.0
    assert float(np.percentile(samples_ms, 95)) < 25.0


@pytest.mark.benchmark
def test_event_detector_latency_budget() -> None:
    detector = EventDetector()
    probs = np.random.rand(9, 1000).astype(np.float32)
    times = np.arange(1000, dtype=np.float64) * 0.005

    samples_ms: list[float] = []
    for _ in range(20):
        detector.reset()
        start = time.perf_counter()
        _ = detector.process_batch(probs, times)
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    assert float(np.mean(samples_ms)) < 15.0
    assert float(np.percentile(samples_ms, 95)) < 25.0


@pytest.mark.benchmark
def test_zmq_one_way_latency_budget() -> None:
    port = _pick_free_port()
    ctx = zmq.Context()
    pub = ZmqPublisher(ctx, port=port, bind=True)
    sub = ZmqSubscriber(ctx, port=port, connect=True)
    time.sleep(0.1)

    samples: list[float] = []
    for _ in range(30):
        send_time = time.monotonic()
        pub.send("heartbeat")
        received = sub.recv(timeout_ms=1000)
        assert received is not None
        recv_time = time.monotonic()
        samples.append((recv_time - send_time) * 1000.0)

    sub.close()
    pub.close()
    ctx.term()

    assert float(np.mean(samples)) < 10.0
    assert float(np.percentile(samples, 95)) < 20.0


@pytest.mark.benchmark
def test_end_to_end_pipeline_latency_budget() -> None:
    config = InferenceConfig(
        checkpoint_path="",
        device="cpu",
        emg_port=_pick_free_port(),
        result_port=_pick_free_port(),
        control_port=_pick_free_port(),
        threshold=0.35,
        rejection_threshold=0.6,
        rest_threshold=0.15,
        debounce_sec=0.05,
        data_timeout_ms=100,
        warm_up_frames=0,
        max_consecutive_drops=2,
    )
    engine = InferenceEngine(config)

    ctx = zmq.Context()
    emg_pub = ZmqPublisher(ctx, port=config.emg_port, bind=True)
    result_sub = ZmqSubscriber(ctx, port=config.result_port, connect=True)
    time.sleep(0.1)

    thread = threading.Thread(target=engine.run, kwargs={"max_messages": 1}, daemon=True)
    thread.start()
    time.sleep(0.1)

    emg_pub.send(
        MsgType.EMG_CHUNK,
        extra_header={"sample_offset": 0},
        array=np.random.randn(16, 40).astype(np.float32),
    )

    pipeline_ms: float | None = None
    deadline = time.time() + 3.0
    while time.time() < deadline:
        received = result_sub.recv(timeout_ms=200)
        if received is None:
            continue
        header, _ = received
        if header.get("msg_type") == MsgType.PROBABILITIES:
            pipeline_ms = float(header["pipeline_ms"])
            break

    thread.join(timeout=3.0)
    if thread.is_alive():
        engine._running = False
        thread.join(timeout=1.0)

    result_sub.close()
    emg_pub.close()
    ctx.term()
    engine.cleanup()

    assert pipeline_ms is not None
    assert pipeline_ms < 200.0
