"""Configuration panel for monitor controls."""

from __future__ import annotations

from dataclasses import asdict

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rt_gesture.config import AppConfig


class ConfigPanel(QWidget):
    """Side panel for config overrides and stream control."""

    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    resume_clicked = pyqtSignal()
    send_shutdown_clicked = pyqtSignal()

    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = config

        layout = QVBoxLayout(self)
        config_group = QGroupBox("Runtime Config")
        form = QFormLayout(config_group)

        self.hdf5_path_edit = QLineEdit(config.data_simulator.hdf5_path)
        browse_hdf5_btn = QPushButton("Browse")
        browse_hdf5_btn.clicked.connect(self._browse_hdf5)
        hdf5_row = QHBoxLayout()
        hdf5_row.addWidget(self.hdf5_path_edit)
        hdf5_row.addWidget(browse_hdf5_btn)
        hdf5_wrap = QWidget()
        hdf5_wrap.setLayout(hdf5_row)
        form.addRow(QLabel("HDF5 Path"), hdf5_wrap)

        self.checkpoint_edit = QLineEdit(config.inference.checkpoint_path)
        browse_ckpt_btn = QPushButton("Browse")
        browse_ckpt_btn.clicked.connect(self._browse_checkpoint)
        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(self.checkpoint_edit)
        ckpt_row.addWidget(browse_ckpt_btn)
        ckpt_wrap = QWidget()
        ckpt_wrap.setLayout(ckpt_row)
        form.addRow(QLabel("Checkpoint"), ckpt_wrap)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(config.inference.threshold)
        form.addRow(QLabel("Threshold"), self.threshold_spin)

        self.rejection_spin = QDoubleSpinBox()
        self.rejection_spin.setRange(0.0, 1.0)
        self.rejection_spin.setSingleStep(0.01)
        self.rejection_spin.setValue(config.inference.rejection_threshold)
        form.addRow(QLabel("Rejection"), self.rejection_spin)

        self.rest_spin = QDoubleSpinBox()
        self.rest_spin.setRange(0.0, 1.0)
        self.rest_spin.setSingleStep(0.01)
        self.rest_spin.setValue(config.inference.rest_threshold)
        form.addRow(QLabel("Rest"), self.rest_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItem("auto", None)
        self.device_combo.addItem("cpu", "cpu")
        self.device_combo.addItem("cuda", "cuda")
        current_device = config.inference.device
        matched_index = 0
        for idx in range(self.device_combo.count()):
            if self.device_combo.itemData(idx) == current_device:
                matched_index = idx
                break
        self.device_combo.setCurrentIndex(matched_index)
        form.addRow(QLabel("Device"), self.device_combo)

        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)
        self.start_btn = QPushButton("Start System")
        self.stop_btn = QPushButton("Stop System")
        self.pause_btn = QPushButton("Pause Display")
        self.resume_btn = QPushButton("Resume Display")
        self.shutdown_btn = QPushButton("Send SHUTDOWN")

        self.start_btn.clicked.connect(self.start_clicked.emit)
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        self.pause_btn.clicked.connect(self.pause_clicked.emit)
        self.resume_btn.clicked.connect(self.resume_clicked.emit)
        self.shutdown_btn.clicked.connect(self.send_shutdown_clicked.emit)

        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.pause_btn)
        controls_layout.addWidget(self.resume_btn)
        controls_layout.addWidget(self.shutdown_btn)

        layout.addWidget(config_group)
        layout.addWidget(controls_group)
        layout.addStretch(1)

    def _browse_hdf5(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select HDF5 File",
            self.hdf5_path_edit.text(),
            "HDF5 Files (*.hdf5 *.h5)",
        )
        if selected:
            self.hdf5_path_edit.setText(selected)

    def _browse_checkpoint(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Checkpoint",
            self.checkpoint_edit.text(),
            "Checkpoint Files (*.ckpt *.pt *.pth)",
        )
        if selected:
            self.checkpoint_edit.setText(selected)

    def apply_to_config(self) -> AppConfig:
        """Return config with panel values applied."""
        self._config.data_simulator.hdf5_path = self.hdf5_path_edit.text().strip()
        self._config.inference.checkpoint_path = self.checkpoint_edit.text().strip()
        self._config.inference.threshold = float(self.threshold_spin.value())
        self._config.inference.rejection_threshold = float(self.rejection_spin.value())
        self._config.inference.rest_threshold = float(self.rest_spin.value())
        self._config.inference.device = self.device_combo.currentData()
        return self._config

    def snapshot(self) -> dict:
        """Return dict snapshot for debugging."""
        return asdict(self.apply_to_config())
