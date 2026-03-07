"""ZMQ transport wrappers with msgpack + optional ndarray payload."""

from __future__ import annotations

import logging
import time
from typing import Any

import msgpack
import numpy as np
import zmq

log = logging.getLogger(__name__)


class ZmqPublisher:
    """Thin publisher wrapper using a unified wire format."""

    def __init__(self, context: zmq.Context, port: int, bind: bool = True) -> None:
        self._socket = context.socket(zmq.PUB)
        if bind:
            self._socket.bind(f"tcp://*:{port}")
        else:
            self._socket.connect(f"tcp://localhost:{port}")
        self._socket.setsockopt(zmq.LINGER, 0)
        self._seq = 0

    def send(
        self,
        msg_type: str,
        extra_header: dict[str, Any] | None = None,
        array: np.ndarray | None = None,
    ) -> None:
        header: dict[str, Any] = {
            "version": 1,
            "seq": self._seq,
            "timestamp": time.monotonic(),
            "msg_type": msg_type,
        }
        if extra_header:
            header.update(extra_header)

        if array is None:
            self._socket.send(msgpack.packb(header, use_bin_type=True))
        else:
            contiguous = np.ascontiguousarray(array)
            header["shape"] = list(contiguous.shape)
            header["dtype"] = str(contiguous.dtype)
            self._socket.send_multipart(
                [
                    msgpack.packb(header, use_bin_type=True),
                    contiguous.tobytes(),
                ]
            )
        self._seq += 1

    def close(self) -> None:
        self._socket.close(linger=0)


class ZmqSubscriber:
    """Thin subscriber wrapper with sequence gap tracking."""

    def __init__(
        self,
        context: zmq.Context,
        port: int,
        connect: bool = True,
        topic: bytes = b"",
    ) -> None:
        self._socket = context.socket(zmq.SUB)
        if connect:
            self._socket.connect(f"tcp://localhost:{port}")
        else:
            self._socket.bind(f"tcp://*:{port}")
        self._socket.setsockopt(zmq.SUBSCRIBE, topic)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._expected_seq = 0
        self.last_gap = 0
        self._port = port

    @property
    def socket(self) -> zmq.Socket:
        return self._socket

    def recv(self, timeout_ms: int = -1) -> tuple[dict[str, Any], np.ndarray | None] | None:
        if timeout_ms >= 0 and not self._socket.poll(timeout_ms, zmq.POLLIN):
            return None

        parts = self._socket.recv_multipart()
        header = msgpack.unpackb(parts[0], raw=False)

        seq = int(header.get("seq", 0))
        self.last_gap = 0
        if seq != self._expected_seq:
            gap = max(0, seq - self._expected_seq)
            self.last_gap = gap
            log.warning(
                "ZMQ port %s seq gap: expected=%s got=%s gap=%s",
                self._port,
                self._expected_seq,
                seq,
                gap,
            )
        self._expected_seq = seq + 1

        array: np.ndarray | None = None
        if len(parts) > 1 and "shape" in header and "dtype" in header:
            dtype = np.dtype(header["dtype"])
            shape = tuple(header["shape"])
            array = np.frombuffer(parts[1], dtype=dtype).reshape(shape).copy()

        return header, array

    def close(self) -> None:
        self._socket.close(linger=0)

