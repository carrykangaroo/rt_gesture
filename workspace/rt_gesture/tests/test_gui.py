from __future__ import annotations

import os

import numpy as np
import pytest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from rt_gesture.config import AppConfig, save_config
from rt_gesture.gui.config_panel import ConfigPanel
from rt_gesture.gui.emg_plot_widget import EmgPlotWidget
from rt_gesture.gui.gesture_display import GestureDisplay
from rt_gesture.gui.main_window import MainWindow
from rt_gesture.gui.status_bar import MonitorStatusBar


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_config_panel_snapshot_roundtrip(qapp: QApplication) -> None:
    config = AppConfig()
    panel = ConfigPanel(config)
    panel.hdf5_path_edit.setText("/tmp/demo.hdf5")
    panel.checkpoint_edit.setText("/tmp/model.ckpt")
    panel.threshold_spin.setValue(0.42)
    panel.rejection_spin.setValue(0.77)
    panel.rest_spin.setValue(0.18)
    panel.device_combo.setCurrentText("cpu")

    snapshot = panel.snapshot()

    assert snapshot["data_simulator"]["hdf5_path"] == "/tmp/demo.hdf5"
    assert snapshot["inference"]["checkpoint_path"] == "/tmp/model.ckpt"
    assert snapshot["inference"]["threshold"] == pytest.approx(0.42, rel=1e-6)
    assert snapshot["inference"]["rejection_threshold"] == pytest.approx(0.77, rel=1e-6)
    assert snapshot["inference"]["rest_threshold"] == pytest.approx(0.18, rel=1e-6)
    assert snapshot["inference"]["device"] == "cpu"


def test_status_bar_update_methods(qapp: QApplication) -> None:
    status = MonitorStatusBar()
    status.update_fps(30.0)
    status.update_latency(12.34)
    status.update_heartbeat("ok")
    status.update_device("cpu")
    status.update_state("Running")

    assert "30.0" in status._fps_label.text()  # type: ignore[attr-defined]
    assert "12.34" in status._latency_label.text()  # type: ignore[attr-defined]
    assert "ok" in status._heartbeat_label.text()  # type: ignore[attr-defined]
    assert "cpu" in status._device_label.text()  # type: ignore[attr-defined]
    assert "Running" in status._state_label.text()  # type: ignore[attr-defined]


def test_gesture_display_probability_heatmap_and_ground_truth(qapp: QApplication) -> None:
    widget = GestureDisplay()
    probs = np.random.rand(9, 25).astype(np.float32)

    widget.update_probabilities(probs)
    widget.append_event("thumb_up", 12.3, 0.91)
    widget.append_ground_truth_prompts(
        [
            {"gesture": "thumb_up", "time": 12.25},
            {"gesture": "index_press", "time": 12.28},
        ]
    )

    assert widget.event_list.count() == 1
    assert widget.gt_list.count() == 2
    assert len(widget._heatmap_buffer) > 0  # type: ignore[attr-defined]
    assert widget.bars[0].value() >= 0


def test_emg_plot_widget_append_and_refresh(qapp: QApplication) -> None:
    widget = EmgPlotWidget(sample_rate=2000, window_sec=2.0)
    widget.resize(320, 240)
    widget.append_chunk(np.random.randn(16, 4000).astype(np.float32))
    widget.refresh()

    x_data, y_data = widget.curves[0].getData()
    assert x_data is not None
    assert y_data is not None
    assert len(x_data) <= 4000


def test_main_window_creates_components_and_timers(qapp: QApplication, tmp_path: Path) -> None:
    config = AppConfig()
    config_path = tmp_path / "gui_config.yaml"
    save_config(config, config_path)

    window = MainWindow(config_path=config_path)
    try:
        assert window.config_panel is not None
        assert window.emg_plot is not None
        assert window.gesture_display is not None
        assert window.monitor_status is not None
        assert window._refresh_timer.isActive()  # type: ignore[attr-defined]
        assert window._heartbeat_timer.isActive()  # type: ignore[attr-defined]
        dumped = window.dump_state_json()
        assert "inference" in dumped
    finally:
        window.close()
