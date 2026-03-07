"""Status bar widget helpers."""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QStatusBar


class MonitorStatusBar(QStatusBar):
    """Simple status bar with runtime stats."""

    def __init__(self) -> None:
        super().__init__()
        self._fps_label = QLabel("FPS: --")
        self._latency_label = QLabel("Latency: -- ms")
        self._heartbeat_label = QLabel("Heartbeat: --")
        self._device_label = QLabel("Device: --")
        self._state_label = QLabel("State: Idle")

        self.addPermanentWidget(self._fps_label)
        self.addPermanentWidget(self._latency_label)
        self.addPermanentWidget(self._heartbeat_label)
        self.addPermanentWidget(self._device_label)
        self.addPermanentWidget(self._state_label)

    def update_fps(self, fps: float) -> None:
        self._fps_label.setText(f"FPS: {fps:.1f}")

    def update_latency(self, latency_ms: float) -> None:
        self._latency_label.setText(f"Latency: {latency_ms:.2f} ms")

    def update_heartbeat(self, status: str) -> None:
        self._heartbeat_label.setText(f"Heartbeat: {status}")

    def update_device(self, device: str) -> None:
        self._device_label.setText(f"Device: {device}")

    def update_state(self, state: str) -> None:
        self._state_label.setText(f"State: {state}")
