"""Gesture probability and event display widgets."""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from rt_gesture.constants import GESTURE_NAMES


class GestureDisplay(QWidget):
    """Display gesture probabilities and detected event log."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        prob_group = QGroupBox("Gesture Probabilities")
        prob_layout = QVBoxLayout(prob_group)
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
            prob_layout.addLayout(row)
            self.bars.append(bar)

        event_group = QGroupBox("Event Log")
        event_layout = QVBoxLayout(event_group)
        self.event_list = QListWidget()
        event_layout.addWidget(self.event_list)

        layout.addWidget(prob_group)
        layout.addWidget(event_group)

    def update_probabilities(self, probs: np.ndarray) -> None:
        """Update bars using latest probability frame."""
        if probs.ndim != 2 or probs.shape[0] != 9 or probs.shape[1] == 0:
            return
        latest = probs[:, -1]
        for idx, bar in enumerate(self.bars):
            value = int(float(latest[idx]) * 1000)
            bar.setValue(max(0, min(1000, value)))

    def append_event(self, gesture: str, event_time: float, confidence: float) -> None:
        self.event_list.addItem(f"{event_time:8.3f}s  {gesture:>14s}  conf={confidence:.3f}")
        if self.event_list.count() > 500:
            self.event_list.takeItem(0)
        self.event_list.scrollToBottom()

