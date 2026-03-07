from __future__ import annotations

import socket
import time

import numpy as np
import zmq

from rt_gesture.zmq_transport import ZmqPublisher, ZmqSubscriber


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_send_recv_numpy_array_round_trip() -> None:
    port = _pick_free_port()
    ctx = zmq.Context()
    pub = ZmqPublisher(ctx, port=port, bind=True)
    sub = ZmqSubscriber(ctx, port=port, connect=True)
    time.sleep(0.05)

    payload = np.random.randn(16, 40).astype(np.float32)
    pub.send("emg_chunk", extra_header={"sample_offset": 0}, array=payload)
    received = sub.recv(timeout_ms=1000)
    assert received is not None
    header, array = received
    assert header["msg_type"] == "emg_chunk"
    assert header["sample_offset"] == 0
    assert np.allclose(array, payload)

    sub.close()
    pub.close()
    ctx.term()


def test_timeout_and_seq_gap_detection() -> None:
    port = _pick_free_port()
    ctx = zmq.Context()
    pub = ZmqPublisher(ctx, port=port, bind=True)
    sub = ZmqSubscriber(ctx, port=port, connect=True)
    time.sleep(0.05)

    assert sub.recv(timeout_ms=20) is None

    pub.send("heartbeat")
    first = sub.recv(timeout_ms=500)
    assert first is not None

    pub._seq += 1  # intentionally introduce a gap for test coverage
    pub.send("heartbeat")
    second = sub.recv(timeout_ms=500)
    assert second is not None
    assert sub.last_gap == 1

    sub.close()
    pub.close()
    ctx.term()

