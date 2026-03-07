"""Gesture probability and event display widgets."""

from __future__ import annotations

from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rt_gesture.constants import GESTURE_NAMES


class GestureDisplay(QWidget):
    """Display gesture probabilities and detected event log."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        bars_group = QGroupBox("Gesture Probabilities (Bars)")
        bars_layout = QVBoxLayout(bars_group)
        self.bars: list[QProgressBar] = []
        for name in GESTURE_NAMES:
            row = QHBoxLayout()
            label = QLabel(name)
            label.setMinimumWidth(120)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(0)
            row.addWidget(label)
            row.addWidget(bar)
            bars_layout.addLayout(row)
            self.bars.append(bar)

        heatmap_group = QGroupBox("Gesture Probabilities (Heatmap)")
        heatmap_layout = QVBoxLayout(heatmap_group)
        self._heatmap_plot = pg.PlotWidget()
        self._heatmap_plot.setLabel("left", "Gesture")
        self._heatmap_plot.setLabel("bottom", "Frames")
        self._heatmap_plot.invertY(True)
        self._heatmap_plot.setYRange(-0.5, len(GESTURE_NAMES) - 0.5, padding=0.0)
        self._heatmap_plot.getAxis("left").setTicks(
            [[(idx, name) for idx, name in enumerate(GESTURE_NAMES)]]
        )
        self._heatmap_image = pg.ImageItem()
        self._heatmap_plot.addItem(self._heatmap_image)
        self._heatmap_image.setLevels((0.0, 1.0))
        lut = pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256)
        self._heatmap_image.setLookupTable(lut)
        heatmap_layout.addWidget(self._heatmap_plot)
        self._heatmap_buffer: deque[np.ndarray] = deque(maxlen=600)

        self._prob_tabs = QTabWidget()
        self._prob_tabs.addTab(bars_group, "Bars")
        self._prob_tabs.addTab(heatmap_group, "Heatmap")

        event_group = QGroupBox("Event Log")
        event_layout = QVBoxLayout(event_group)
        self.event_list = QListWidget()
        event_layout.addWidget(self.event_list)

        gt_group = QGroupBox("Ground Truth Prompts")
        gt_layout = QVBoxLayout(gt_group)
        self.gt_list = QListWidget()
        gt_layout.addWidget(self.gt_list)

        layout.addWidget(self._prob_tabs)
        layout.addWidget(event_group)
        layout.addWidget(gt_group)

    def update_probabilities(self, probs: np.ndarray) -> None:
        """Update bars using latest probability frame."""
        if probs.ndim != 2 or probs.shape[0] != 9 or probs.shape[1] == 0:
            return
        latest = probs[:, -1]
        for idx, bar in enumerate(self.bars):
            value = int(float(latest[idx]) * 1000)
            bar.setValue(max(0, min(1000, value)))

        for frame in probs.T:
            self._heatmap_buffer.append(frame.astype(np.float32))
        if self._heatmap_buffer:
            image = np.stack(list(self._heatmap_buffer), axis=1)
            self._heatmap_image.setImage(image, autoLevels=False)

    def append_event(self, gesture: str, event_time: float, confidence: float) -> None:
        self.event_list.addItem(f"{event_time:8.3f}s  {gesture:>14s}  conf={confidence:.3f}")
        if self.event_list.count() > 500:
            self.event_list.takeItem(0)
        self.event_list.scrollToBottom()

    def append_ground_truth(self, gesture: str, event_time: float) -> None:
        self.gt_list.addItem(f"{event_time:8.3f}s  {gesture:>14s}")
        if self.gt_list.count() > 500:
            self.gt_list.takeItem(0)
        self.gt_list.scrollToBottom()

    def append_ground_truth_prompts(self, prompts: list[dict[str, float | str]]) -> None:
        for prompt in prompts:
            gesture = str(prompt.get("gesture", "unknown"))
            event_time = float(prompt.get("time", 0.0))
            self.append_ground_truth(gesture, event_time)
