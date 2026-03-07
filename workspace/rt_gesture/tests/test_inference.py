from __future__ import annotations

import logging
import socket
import threading
import time

import numpy as np
import torch
import zmq

from rt_gesture.config import InferenceConfig
from rt_gesture.constants import MsgType
from rt_gesture.event_detector import GestureEvent
from rt_gesture.inference_engine import InferenceEngine
from rt_gesture.zmq_transport import ZmqPublisher, ZmqSubscriber


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_inference_config() -> InferenceConfig:
    return InferenceConfig(
        checkpoint_path="",
        device="cpu",
        emg_port=_pick_free_port(),
        result_port=_pick_free_port(),
        control_port=_pick_free_port(),
        threshold=0.2,
        rejection_threshold=0.2,
        rest_threshold=0.05,
        debounce_sec=0.01,
        data_timeout_ms=100,
        warm_up_frames=7,
        max_consecutive_drops=2,
    )


def test_data_interruption_resets_state() -> None:
    config = _build_inference_config()
    engine = InferenceEngine(config)
    try:
        engine._next_frame_index = 42
        engine._conv_history = torch.zeros((1, 16, 10), dtype=torch.float32)
        engine.event_detector.process_frame(np.full(9, 0.9), timestamp=0.2)

        engine._handle_data_interruption(reason="unit-test")

        assert engine._warm_up_remaining == config.warm_up_frames
        assert engine._conv_history.shape == (1, 16, 0)
        assert engine._lstm_state is None
        assert engine.event_detector._prev_probs is None  # type: ignore[attr-defined]
    finally:
        engine.cleanup()


def test_control_shutdown_stops_engine_loop() -> None:
    config = _build_inference_config()
    engine = InferenceEngine(config)

    thread = threading.Thread(target=engine.run, kwargs={"max_messages": None}, daemon=True)
    thread.start()
    time.sleep(0.1)

    ctx = zmq.Context()
    publisher = ZmqPublisher(ctx, port=config.control_port, bind=True)
    time.sleep(0.1)
    publisher.send(MsgType.SHUTDOWN)

    thread.join(timeout=3)
    publisher.close()
    ctx.term()

    assert not thread.is_alive()
    engine.cleanup()


def test_inference_emits_heartbeat() -> None:
    config = _build_inference_config()
    config.heartbeat_interval_sec = 0.05
    config.data_timeout_ms = 50
    engine = InferenceEngine(config)

    ctx = zmq.Context()
    result_sub = ZmqSubscriber(ctx, port=config.result_port, connect=True)
    control_pub = ZmqPublisher(ctx, port=config.control_port, bind=True)
    time.sleep(0.1)

    thread = threading.Thread(target=engine.run, kwargs={"max_messages": None}, daemon=True)
    thread.start()

    heartbeat_header: dict | None = None
    deadline = time.time() + 3.0
    while time.time() < deadline:
        received = result_sub.recv(timeout_ms=200)
        if received is None:
            continue
        header, _ = received
        if header.get("msg_type") == MsgType.HEARTBEAT:
            heartbeat_header = header
            break

    control_pub.send(MsgType.SHUTDOWN)
    thread.join(timeout=3.0)
    if thread.is_alive():
        engine._running = False
        thread.join(timeout=1.0)
    assert not thread.is_alive()

    control_pub.close()
    result_sub.close()
    ctx.term()
    engine.cleanup()

    assert heartbeat_header is not None
    assert heartbeat_header["msg_type"] == MsgType.HEARTBEAT
    assert "device" in heartbeat_header


def test_inference_latency_headers_and_callback_api() -> None:
    config = _build_inference_config()
    config.threshold = 0.0
    config.rejection_threshold = 0.0
    config.rest_threshold = 0.0
    config.debounce_sec = 0.0
    config.warm_up_frames = 0
    config.max_messages = 1
    engine = InferenceEngine(config)

    callback_events: list[GestureEvent] = []

    def _on_event(event: GestureEvent) -> None:
        callback_events.append(event)

    engine.register_event_callback(_on_event)

    ctx = zmq.Context()
    emg_pub = ZmqPublisher(ctx, port=config.emg_port, bind=True)
    result_sub = ZmqSubscriber(ctx, port=config.result_port, connect=True)
    time.sleep(0.1)

    thread = threading.Thread(target=engine.run, kwargs={"max_messages": 1}, daemon=True)
    thread.start()
    time.sleep(0.1)

    emg = np.random.randn(16, 40).astype(np.float32)
    emg_pub.send(
        MsgType.EMG_CHUNK,
        extra_header={"sample_offset": 0, "stream_start_time": 10.0},
        array=emg,
    )

    saw_probabilities = False
    saw_event = False
    deadline = time.time() + 3.0
    while time.time() < deadline and (not saw_probabilities or not saw_event):
        received = result_sub.recv(timeout_ms=200)
        if received is None:
            continue
        header, array = received
        msg_type = header.get("msg_type")
        if msg_type == MsgType.PROBABILITIES:
            saw_probabilities = True
            assert array is not None
            for key in ("transport_ms", "infer_ms", "pipeline_ms", "time_start_rel", "time_end_rel"):
                assert key in header
                assert float(header[key]) >= 0.0
            assert float(header["time_start"]) >= 10.0
        if msg_type == MsgType.GESTURE_EVENT:
            saw_event = True
            for key in ("transport_ms", "infer_ms", "post_ms", "pipeline_ms", "event_time_rel"):
                assert key in header
                assert float(header[key]) >= 0.0
            assert float(header["event_time"]) >= 10.0

    thread.join(timeout=3.0)
    if thread.is_alive():
        engine._running = False
        thread.join(timeout=1.0)
    assert not thread.is_alive()

    engine.unregister_event_callback(_on_event)
    callback_count = len(callback_events)
    engine._notify_event_callbacks(
        GestureEvent(gesture="thumb_up", timestamp=0.123, confidence=0.99)
    )

    emg_pub.close()
    result_sub.close()
    ctx.term()
    engine.cleanup()

    assert saw_probabilities
    assert saw_event
    assert callback_events
    assert len(callback_events) == callback_count


def test_latency_warning_thresholds_emit_warning_logs(caplog) -> None:
    config = _build_inference_config()
    config.latency_warn_transport_ms = 0.1
    config.latency_warn_infer_ms = 0.1
    config.latency_warn_post_ms = 0.1
    config.latency_warn_pipeline_ms = 0.1
    engine = InferenceEngine(config)
    try:
        with caplog.at_level(logging.WARNING):
            engine._warn_latency_if_needed(
                transport_ms=1.0,
                infer_ms=1.0,
                post_ms=1.0,
                pipeline_ms=1.0,
            )
        assert "Transport latency high" in caplog.text
        assert "Inference latency high" in caplog.text
        assert "Post-process latency high" in caplog.text
        assert "Pipeline latency high" in caplog.text
    finally:
        engine.cleanup()
