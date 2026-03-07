"""HDF5 data simulator that replays EMG samples in real-time cadence."""

from __future__ import annotations

import atexit
import logging
import signal
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import zmq

from rt_gesture.config import DataSimulatorConfig
from rt_gesture.constants import EMG_SAMPLE_RATE, MsgType
from rt_gesture.zmq_transport import ZmqPublisher, ZmqSubscriber

log = logging.getLogger(__name__)


class DataSimulator:
    """Replay EMG data from an HDF5 recording with real-time timing."""

    def __init__(self, config: DataSimulatorConfig) -> None:
        self.config = config
        self.chunk_size = config.chunk_size
        self.chunk_interval_sec = self.chunk_size / EMG_SAMPLE_RATE

        self._ctx = zmq.Context()
        self._emg_pub = ZmqPublisher(self._ctx, config.emg_port)
        self._gt_pub = ZmqPublisher(self._ctx, config.gt_port)
        self._ctrl_sub = ZmqSubscriber(self._ctx, config.control_port)

        self._hdf5_file: h5py.File | None = None
        self._data: np.ndarray | None = None
        self._prompts: pd.DataFrame = pd.DataFrame(columns=["name", "time"])
        self._stream_start_time: float | None = None
        self._running = False

        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, _frame: object) -> None:
        log.info("DataSimulator received signal %s", signum)
        self._running = False

    def load_data(self, hdf5_path: str | Path) -> None:
        path = Path(hdf5_path)
        if not path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {path}")

        self._hdf5_file = h5py.File(path, "r")
        self._data = self._hdf5_file["data"][:]
        if "time" not in self._data.dtype.fields or "emg" not in self._data.dtype.fields:
            raise ValueError("HDF5 dataset /data must contain fields: emg, time")

        try:
            prompts = pd.read_hdf(path, "prompts")
        except (KeyError, FileNotFoundError, OSError, ValueError):
            prompts = pd.DataFrame(columns=["name", "time"])
        self._prompts = prompts

        if len(self._data) > 0:
            self._stream_start_time = float(self._data["time"][0])
        log.info(
            "Loaded HDF5 file %s with %d samples and %d prompts",
            path,
            len(self._data),
            len(self._prompts),
        )

    def run(self, max_chunks: int | None = None) -> None:
        if self._data is None:
            raise RuntimeError("load_data() must be called before run()")

        total_samples = len(self._data)
        sample_offset = 0
        chunks_sent = 0
        self._running = True
        start_mono = time.monotonic()
        log.info("DataSimulator started")

        while self._running and sample_offset < total_samples:
            control_msg = self._ctrl_sub.recv(timeout_ms=0)
            if control_msg is not None:
                header, _ = control_msg
                if header.get("msg_type") == MsgType.SHUTDOWN:
                    log.info("DataSimulator received SHUTDOWN")
                    break

            end_offset = min(sample_offset + self.chunk_size, total_samples)
            chunk = self._data[sample_offset:end_offset]
            if len(chunk) == 0:
                break

            emg = np.stack(chunk["emg"], axis=0).T.astype(np.float32)
            timestamps = chunk["time"]
            self._emg_pub.send(
                MsgType.EMG_CHUNK,
                extra_header={
                    "sample_offset": int(sample_offset),
                    "stream_start_time": float(self._stream_start_time or 0.0),
                },
                array=emg,
            )

            if not self._prompts.empty:
                t_start = float(timestamps[0])
                t_end = float(timestamps[-1])
                prompts_in_range = self._prompts[self._prompts["time"].between(t_start, t_end)]
                if len(prompts_in_range) > 0:
                    serialized = [
                        {"gesture": str(row["name"]), "time": float(row["time"])}
                        for _, row in prompts_in_range.iterrows()
                    ]
                    self._gt_pub.send(
                        MsgType.GROUND_TRUTH,
                        extra_header={"prompts": serialized},
                    )

            sample_offset = end_offset
            chunks_sent += 1
            if max_chunks is not None and chunks_sent >= max_chunks:
                break

            target_time = start_mono + sample_offset / EMG_SAMPLE_RATE
            sleep_duration = target_time - time.monotonic()
            if sleep_duration > 0:
                time.sleep(sleep_duration)

        log.info(
            "DataSimulator stopped after %d chunks (%d/%d samples)",
            chunks_sent,
            sample_offset,
            total_samples,
        )
        self.cleanup()

    def cleanup(self) -> None:
        self._running = False
        if self._hdf5_file is not None:
            self._hdf5_file.close()
            self._hdf5_file = None
        if hasattr(self, "_emg_pub"):
            self._emg_pub.close()
        if hasattr(self, "_gt_pub"):
            self._gt_pub.close()
        if hasattr(self, "_ctrl_sub"):
            self._ctrl_sub.close()
        if hasattr(self, "_ctx"):
            self._ctx.term()

