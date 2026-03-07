"""Main GUI window for RT-Gesture monitor."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import QDockWidget, QMainWindow, QSplitter

import zmq

from rt_gesture.config import AppConfig, load_config, save_config
from rt_gesture.constants import MsgType
from rt_gesture.gui.config_panel import ConfigPanel
from rt_gesture.gui.emg_plot_widget import EmgPlotWidget
from rt_gesture.gui.gesture_display import GestureDisplay
from rt_gesture.gui.process_manager import ProcessManager
from rt_gesture.gui.status_bar import MonitorStatusBar
from rt_gesture.zmq_transport import ZmqSubscriber

log = logging.getLogger(__name__)


class ZmqReaderThread(QThread):
    """Read EMG and result streams in a dedicated thread."""

    emg_chunk = pyqtSignal(object)
    probability_frame = pyqtSignal(object, object)
    gesture_event = pyqtSignal(dict)
    ground_truth = pyqtSignal(object)
    heartbeat = pyqtSignal(dict)
    stream_status = pyqtSignal(str)

    def __init__(self, app_config: AppConfig) -> None:
        super().__init__()
        self._cfg = app_config
        self._running = False

    def run(self) -> None:
        self._running = True
        ctx = zmq.Context()
        emg_sub = ZmqSubscriber(ctx, port=self._cfg.gui.emg_port, connect=True)
        gt_sub = ZmqSubscriber(ctx, port=self._cfg.gui.gt_port, connect=True)
        result_sub = ZmqSubscriber(ctx, port=self._cfg.gui.result_port, connect=True)
        poller = zmq.Poller()
        poller.register(emg_sub.socket, zmq.POLLIN)
        poller.register(gt_sub.socket, zmq.POLLIN)
        poller.register(result_sub.socket, zmq.POLLIN)
        self.stream_status.emit("Connected")

        try:
            while self._running:
                events = dict(poller.poll(timeout=100))
                if emg_sub.socket in events:
                    received = emg_sub.recv(timeout_ms=0)
                    if received is not None:
                        _, array = received
                        if array is not None:
                            self.emg_chunk.emit(array)
                if gt_sub.socket in events:
                    received = gt_sub.recv(timeout_ms=0)
                    if received is not None:
                        header, _ = received
                        if header.get("msg_type") == MsgType.GROUND_TRUTH:
                            self.ground_truth.emit(header.get("prompts", []))
                if result_sub.socket in events:
                    received = result_sub.recv(timeout_ms=0)
                    if received is None:
                        continue
                    header, array = received
                    msg_type = header.get("msg_type")
                    if msg_type == MsgType.PROBABILITIES and array is not None:
                        self.probability_frame.emit(header, array)
                    elif msg_type == MsgType.GESTURE_EVENT:
                        self.gesture_event.emit(header)
                    elif msg_type == MsgType.HEARTBEAT:
                        self.heartbeat.emit(header)
        finally:
            emg_sub.close()
            gt_sub.close()
            result_sub.close()
            ctx.term()
            self.stream_status.emit("Disconnected")

    def stop(self) -> None:
        self._running = False
        self.wait(1000)


class MainWindow(QMainWindow):
    """Top-level monitor UI."""

    def __init__(self, config_path: str | Path = "config/default.yaml") -> None:
        super().__init__()
        self.setWindowTitle("RT-Gesture Monitor")
        self.resize(1400, 900)
        self._config_path = Path(config_path)
        self._config = load_config(self._config_path)
        self._reader_thread: ZmqReaderThread | None = None
        self._process_manager = ProcessManager()
        self._display_paused = False

        self._frame_counter = 0
        self._fps_started_at = time.monotonic()
        self._last_heartbeat_at = 0.0

        self.emg_plot = EmgPlotWidget(window_sec=self._config.gui.plot_duration_sec)
        self.gesture_display = GestureDisplay()

        splitter = QSplitter()
        splitter.addWidget(self.emg_plot)
        splitter.addWidget(self.gesture_display)
        splitter.setSizes([800, 400])
        self.setCentralWidget(splitter)

        self.config_panel = ConfigPanel(self._config)
        self.config_panel.start_clicked.connect(self.start_monitor)
        self.config_panel.stop_clicked.connect(self.stop_monitor)
        self.config_panel.pause_clicked.connect(self.pause_display)
        self.config_panel.resume_clicked.connect(self.resume_display)
        self.config_panel.send_shutdown_clicked.connect(self.send_shutdown)

        dock = QDockWidget("Config", self)
        dock.setWidget(self.config_panel)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        self.monitor_status = MonitorStatusBar()
        self.setStatusBar(self.monitor_status)
        self.monitor_status.update_device(self._config.inference.device or "auto")
        self.monitor_status.update_heartbeat("--")

        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self.emg_plot.refresh)
        self._refresh_timer.start(int(1000 / max(1, self._config.gui.refresh_rate_hz)))

        self._heartbeat_timer = QTimer(self)
        self._heartbeat_timer.timeout.connect(self._check_heartbeat_timeout)
        self._heartbeat_timer.start(300)

    def _attach_reader_signals(self, reader: ZmqReaderThread) -> None:
        reader.emg_chunk.connect(self.on_emg_chunk)
        reader.probability_frame.connect(self.on_probability_frame)
        reader.gesture_event.connect(self.on_gesture_event)
        reader.ground_truth.connect(self.on_ground_truth)
        reader.heartbeat.connect(self.on_heartbeat)
        reader.stream_status.connect(self.monitor_status.update_state)

    def start_monitor(self) -> None:
        self._config = self.config_panel.apply_to_config()
        save_config(self._config, self._config_path)
        if not self._process_manager.is_running():
            self._process_manager.start(self._config_path)
            self.monitor_status.update_state("Starting backend")
            self.monitor_status.update_device(self._config.inference.device or "auto")
            time.sleep(0.2)
        if self._reader_thread is not None and self._reader_thread.isRunning():
            return
        self._reader_thread = ZmqReaderThread(self._config)
        self._attach_reader_signals(self._reader_thread)
        self._reader_thread.start()
        self._last_heartbeat_at = time.monotonic()
        self.monitor_status.update_heartbeat("waiting")
        self.monitor_status.update_state("Running")

    def stop_monitor(self) -> None:
        if self._reader_thread is not None:
            self._reader_thread.stop()
            self._reader_thread = None
        self._process_manager.stop(
            control_port=self._config.gui.control_port,
            timeout_sec=self._config.shutdown_timeout_sec,
        )
        self.monitor_status.update_heartbeat("--")
        self.monitor_status.update_state("Stopped")

    def pause_display(self) -> None:
        self._display_paused = True
        self.monitor_status.update_state("Paused")

    def resume_display(self) -> None:
        self._display_paused = False
        self.monitor_status.update_state("Monitoring")

    def send_shutdown(self) -> None:
        self._process_manager.request_shutdown(self._config.gui.control_port)
        self.monitor_status.update_state("SHUTDOWN sent")

    def on_emg_chunk(self, emg_array: object) -> None:
        if self._display_paused:
            return
        self.emg_plot.append_chunk(emg_array)  # type: ignore[arg-type]

    def on_probability_frame(self, header: object, probs: object) -> None:
        if self._display_paused:
            return
        self.gesture_display.update_probabilities(probs)  # type: ignore[arg-type]
        header_dict = header if isinstance(header, dict) else {}

        self._frame_counter += 1
        elapsed = time.monotonic() - self._fps_started_at
        if elapsed >= 1.0:
            fps = self._frame_counter / elapsed
            self.monitor_status.update_fps(fps)
            self._frame_counter = 0
            self._fps_started_at = time.monotonic()

        infer_ms = float(header_dict.get("infer_ms", 0.0))
        transport_ms = float(header_dict.get("transport_ms", 0.0))
        self.monitor_status.update_latency(infer_ms + transport_ms)

    def on_gesture_event(self, header: object) -> None:
        if self._display_paused:
            return
        header_dict = header if isinstance(header, dict) else {}
        self.gesture_display.append_event(
            str(header_dict.get("gesture", "unknown")),
            float(header_dict.get("event_time", 0.0)),
            float(header_dict.get("confidence", 0.0)),
        )

    def on_ground_truth(self, prompts: object) -> None:
        if self._display_paused:
            return
        if isinstance(prompts, list):
            self.gesture_display.append_ground_truth_prompts(prompts)

    def on_heartbeat(self, header: object) -> None:
        header_dict = header if isinstance(header, dict) else {}
        self._last_heartbeat_at = time.monotonic()
        self.monitor_status.update_heartbeat("ok")
        if "device" in header_dict:
            self.monitor_status.update_device(str(header_dict["device"]))

    def _check_heartbeat_timeout(self) -> None:
        if self._reader_thread is None or not self._reader_thread.isRunning():
            return
        if self._last_heartbeat_at <= 0:
            return
        elapsed = time.monotonic() - self._last_heartbeat_at
        if elapsed > self._config.gui.heartbeat_timeout_sec:
            self.monitor_status.update_heartbeat("timeout")
            self.monitor_status.update_state("Heartbeat timeout")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.stop_monitor()
        super().closeEvent(event)

    def dump_state_json(self) -> str:
        return json.dumps(self.config_panel.snapshot(), ensure_ascii=False, indent=2)
