"""Streaming inference engine process."""

from __future__ import annotations

import atexit
import logging
import signal
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import zmq

from rt_gesture.checkpoint_utils import auto_select_device, load_model_from_lightning_checkpoint
from rt_gesture.config import InferenceConfig
from rt_gesture.constants import EMG_NUM_CHANNELS, EMG_SAMPLE_RATE, MODEL_STRIDE, MsgType
from rt_gesture.event_detector import EventDetector, EventDetectorConfig, GestureEvent
from rt_gesture.logger import EventLogger, PredictionStore
from rt_gesture.networks import DiscreteGesturesArchitecture
from rt_gesture.zmq_transport import ZmqPublisher, ZmqSubscriber

log = logging.getLogger(__name__)


class InferenceEngine:
    """Receive EMG chunks and publish probabilities/events."""

    def __init__(self, config: InferenceConfig, run_dir: str | Path | None = None) -> None:
        self.config = config
        self.device = config.device or auto_select_device()

        checkpoint_path = Path(config.checkpoint_path) if config.checkpoint_path else None
        if checkpoint_path and checkpoint_path.exists():
            self.model = load_model_from_lightning_checkpoint(checkpoint_path, device=self.device)
        else:
            log.warning("Checkpoint missing, using randomly initialized model")
            self.model = DiscreteGesturesArchitecture().to(self.device).eval()

        self._ctx = zmq.Context()
        self._emg_sub = ZmqSubscriber(self._ctx, config.emg_port)
        self._result_pub = ZmqPublisher(self._ctx, config.result_port)
        self._ctrl_sub = ZmqSubscriber(self._ctx, config.control_port)
        self._ctrl_pub = ZmqPublisher(self._ctx, config.control_port, bind=False)

        self._running = False
        self._consecutive_drops = 0
        self._next_frame_index = 0
        self._has_received_data = False
        self._data_interrupted = False
        self._last_heartbeat_time = 0.0
        self._event_callbacks: list[Callable[[GestureEvent], None]] = []

        self.event_detector = EventDetector(
            EventDetectorConfig(
                threshold=config.threshold,
                rejection_threshold=config.rejection_threshold,
                rest_threshold=config.rest_threshold,
                debounce_sec=config.debounce_sec,
            )
        )
        self._reset_model_state()

        self._event_logger: EventLogger | None = None
        self._prediction_store: PredictionStore | None = None
        if run_dir is not None:
            self._event_logger = EventLogger(run_dir)
            self._prediction_store = PredictionStore(run_dir)

        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, _frame: object) -> None:
        log.info("InferenceEngine received signal %s", signum)
        self._running = False

    def register_event_callback(self, callback: Callable[[GestureEvent], None]) -> None:
        """Register a callback invoked when a gesture event is detected."""
        self._event_callbacks.append(callback)

    def unregister_event_callback(self, callback: Callable[[GestureEvent], None]) -> None:
        """Unregister a previously registered event callback."""
        self._event_callbacks = [fn for fn in self._event_callbacks if fn is not callback]

    def _notify_event_callbacks(self, event: GestureEvent) -> None:
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception:
                log.exception("Event callback raised an exception")

    def _emit_heartbeat_if_due(self) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat_time < self.config.heartbeat_interval_sec:
            return
        self._result_pub.send(
            MsgType.HEARTBEAT,
            extra_header={"device": self.device},
        )
        self._ctrl_pub.send(
            MsgType.HEARTBEAT,
            extra_header={"device": self.device},
        )
        self._last_heartbeat_time = now

    def _warn_latency_if_needed(
        self,
        transport_ms: float,
        infer_ms: float,
        post_ms: float,
        pipeline_ms: float,
    ) -> None:
        if transport_ms > self.config.latency_warn_transport_ms:
            log.warning(
                "Transport latency high: %.3f ms (threshold %.3f ms)",
                transport_ms,
                self.config.latency_warn_transport_ms,
            )
        if infer_ms > self.config.latency_warn_infer_ms:
            log.warning(
                "Inference latency high: %.3f ms (threshold %.3f ms)",
                infer_ms,
                self.config.latency_warn_infer_ms,
            )
        if post_ms > self.config.latency_warn_post_ms:
            log.warning(
                "Post-process latency high: %.3f ms (threshold %.3f ms)",
                post_ms,
                self.config.latency_warn_post_ms,
            )
        if pipeline_ms > self.config.latency_warn_pipeline_ms:
            log.warning(
                "Pipeline latency high: %.3f ms (threshold %.3f ms)",
                pipeline_ms,
                self.config.latency_warn_pipeline_ms,
            )

    def _reset_model_state(self) -> None:
        self._conv_history = torch.zeros(
            1,
            EMG_NUM_CHANNELS,
            0,
            device=self.device,
            dtype=torch.float32,
        )
        self._lstm_state: tuple[torch.Tensor, torch.Tensor] | None = None
        self._warm_up_remaining = 0

    def _handle_data_interruption(self, reason: str) -> None:
        log.warning("Data interruption detected (%s), resetting states", reason)
        self._reset_model_state()
        self._warm_up_remaining = self.config.warm_up_frames
        self.event_detector.reset()
        self._consecutive_drops = 0
        self._data_interrupted = True

    def run(self, max_messages: int | None = None) -> None:
        self._running = True
        processed_messages = 0
        poller = zmq.Poller()
        poller.register(self._emg_sub.socket, zmq.POLLIN)
        poller.register(self._ctrl_sub.socket, zmq.POLLIN)
        log.info("InferenceEngine started on device=%s", self.device)

        while self._running:
            self._emit_heartbeat_if_due()
            events = dict(poller.poll(timeout=self.config.data_timeout_ms))

            if self._ctrl_sub.socket in events:
                control = self._ctrl_sub.recv(timeout_ms=0)
                if control is not None:
                    header, _ = control
                    if header.get("msg_type") == MsgType.SHUTDOWN:
                        log.info("InferenceEngine received SHUTDOWN")
                        self._ctrl_pub.send(MsgType.SHUTDOWN)
                        break

            if self._emg_sub.socket not in events:
                if self._has_received_data and not self._data_interrupted:
                    self._handle_data_interruption(reason="timeout")
                continue

            payload = self._emg_sub.recv(timeout_ms=0)
            if payload is None:
                continue
            header, emg_array = payload
            if emg_array is None:
                continue
            self._has_received_data = True
            self._data_interrupted = False

            if self._emg_sub.last_gap > 0:
                self._consecutive_drops += self._emg_sub.last_gap
            else:
                self._consecutive_drops = 0
            if self._consecutive_drops >= self.config.max_consecutive_drops:
                self._handle_data_interruption(reason="sequence gap")
                continue

            recv_time = time.monotonic()
            transport_ms = (recv_time - float(header["timestamp"])) * 1000.0

            emg_tensor = torch.from_numpy(emg_array).unsqueeze(0).to(self.device, dtype=torch.float32)

            infer_start = time.monotonic()
            with torch.no_grad():
                logits, self._conv_history, self._lstm_state = self.model.forward_streaming(
                    new_samples=emg_tensor,
                    conv_history=self._conv_history,
                    lstm_state=self._lstm_state,
                )
            infer_ms = (time.monotonic() - infer_start) * 1000.0
            frame_count = logits.shape[2]
            if frame_count == 0:
                processed_messages += 1
                if max_messages is not None and processed_messages >= max_messages:
                    break
                continue

            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy().astype(np.float32)
            frame_times = (
                self._next_frame_index + np.arange(frame_count, dtype=np.float64)
            ) * (MODEL_STRIDE / EMG_SAMPLE_RATE)
            self._next_frame_index += frame_count

            if self._warm_up_remaining > 0:
                skip = min(self._warm_up_remaining, frame_count)
                probs = probs[:, skip:]
                frame_times = frame_times[skip:]
                self._warm_up_remaining -= skip
                if probs.shape[1] == 0:
                    processed_messages += 1
                    if max_messages is not None and processed_messages >= max_messages:
                        break
                    continue

            self._result_pub.send(
                MsgType.PROBABILITIES,
                extra_header={
                    "time_start": float(frame_times[0]),
                    "time_end": float(frame_times[-1]),
                    "infer_ms": round(infer_ms, 3),
                    "transport_ms": round(transport_ms, 3),
                    "pipeline_ms": round((time.monotonic() - float(header["timestamp"])) * 1000.0, 3),
                },
                array=probs,
            )

            if self._prediction_store is not None:
                self._prediction_store.append(probs, frame_times)

            post_start = time.monotonic()
            events = self.event_detector.process_batch(probs, frame_times)
            post_ms = (time.monotonic() - post_start) * 1000.0
            base_pipeline_ms = (time.monotonic() - float(header["timestamp"])) * 1000.0
            self._warn_latency_if_needed(
                transport_ms=transport_ms,
                infer_ms=infer_ms,
                post_ms=post_ms,
                pipeline_ms=base_pipeline_ms,
            )

            for event in events:
                pipeline_ms = (time.monotonic() - float(header["timestamp"])) * 1000.0
                self._result_pub.send(
                    MsgType.GESTURE_EVENT,
                    extra_header={
                        "gesture": event.gesture,
                        "event_time": event.timestamp,
                        "confidence": event.confidence,
                        "transport_ms": round(transport_ms, 3),
                        "infer_ms": round(infer_ms, 3),
                        "post_ms": round(post_ms, 3),
                        "pipeline_ms": round(pipeline_ms, 3),
                    },
                )
                self._notify_event_callbacks(event)
                if self._event_logger is not None:
                    self._event_logger.log_event(event.gesture, event.timestamp, event.confidence)
                log.info(
                    "Detected %s at %.3fs conf=%.3f pipeline=%.3fms",
                    event.gesture,
                    event.timestamp,
                    event.confidence,
                    pipeline_ms,
                )

            processed_messages += 1
            if max_messages is not None and processed_messages >= max_messages:
                break

        self.cleanup()

    def cleanup(self) -> None:
        self._running = False
        if hasattr(self, "_event_logger") and self._event_logger is not None:
            self._event_logger.close()
            self._event_logger = None
        if hasattr(self, "_prediction_store") and self._prediction_store is not None:
            self._prediction_store.save()
            self._prediction_store = None
        if hasattr(self, "_emg_sub"):
            self._emg_sub.close()
        if hasattr(self, "_result_pub"):
            self._result_pub.close()
        if hasattr(self, "_ctrl_sub"):
            self._ctrl_sub.close()
        if hasattr(self, "_ctrl_pub"):
            self._ctrl_pub.close()
        if hasattr(self, "_ctx"):
            self._ctx.term()

        if hasattr(self, "model"):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
