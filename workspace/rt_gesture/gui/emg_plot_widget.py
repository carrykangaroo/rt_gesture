"""Real-time EMG plot widget."""

from __future__ import annotations

from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class EmgPlotWidget(QWidget):
    """Display 16-channel EMG stream with channel offsets."""

    def __init__(self, sample_rate: int = 2000, window_sec: float = 5.0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.sample_rate = sample_rate
        self.window_sec = window_sec
        self.max_samples = int(sample_rate * window_sec)

        self._buffer: deque[np.ndarray] = deque()
        self._buffer_len = 0

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("left", "Channels")
        self.plot.setLabel("bottom", "Samples")

        layout = QVBoxLayout(self)
        layout.addWidget(self.plot)

        self.curves = []
        for idx in range(16):
            color = pg.intColor(idx, hues=16)
            curve = self.plot.plot(pen=pg.mkPen(color=color, width=1))
            self.curves.append(curve)

    def append_chunk(self, emg: np.ndarray) -> None:
        """Append new EMG data chunk with shape (16, N)."""
        if emg.ndim != 2 or emg.shape[0] != 16:
            return
        self._buffer.append(emg)
        self._buffer_len += emg.shape[1]

        while self._buffer and self._buffer_len > self.max_samples:
            removed = self._buffer.popleft()
            self._buffer_len -= removed.shape[1]

    def refresh(self) -> None:
        """Redraw plot from buffered data."""
        if not self._buffer:
            return
        data = np.concatenate(list(self._buffer), axis=1)
        n = data.shape[1]
        x = np.arange(n, dtype=np.float32)
        view_width_px = max(1, int(self.plot.width()))
        downsample_step = max(1, n // view_width_px) if n > view_width_px else 1
        if downsample_step > 1:
            x = x[::downsample_step]

        for channel_idx, curve in enumerate(self.curves):
            y = data[channel_idx] + channel_idx * 250.0
            if downsample_step > 1:
                y = y[::downsample_step]
            curve.setData(x, y)
