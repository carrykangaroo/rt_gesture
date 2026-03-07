"""Dataset utilities for RT-Gesture training and evaluation.

Internalized from generic_neuromotor_interface.data / .utils to eliminate
external dependency.  Handwriting-specific classes have been pruned.

Original: Copyright (c) Meta Platforms, Inc. and affiliates. (LICENSE applies)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm
from typing_extensions import Self

from rt_gesture.constants import EMG_SAMPLE_RATE
from rt_gesture.transforms import DiscreteGesturesTransform, RotationAugmentation, Transform

# ------------------------------------------------------------------
# Utilities  (from utils.py)
# ------------------------------------------------------------------

def get_full_dataset_path(root: str, dataset: str) -> Path:
    """Return ``root / dataset.hdf5``."""
    return Path(root) / f"{dataset}.hdf5"


# ------------------------------------------------------------------
# Partitions type
# ------------------------------------------------------------------
Partitions = list[tuple[float, float]] | None


# ------------------------------------------------------------------
# DataSplit
# ------------------------------------------------------------------
@dataclass
class DataSplit:
    """Train, val, and test datasets, with partitions to sample within each dataset."""

    train: dict[str, Partitions | None]
    val: dict[str, Partitions | None]
    test: dict[str, Partitions | None]

    @classmethod
    def from_csv(
        cls, csv_filename: str, pool_test_partitions: bool = False
    ) -> "DataSplit":
        """Create splits from csv file with (dataset, start, end, split) columns."""
        df = pd.read_csv(csv_filename)
        splits: dict[str, dict] = {}

        for split in ["train", "val", "test"]:
            splits[split] = {}
            for dataset in df[df["split"] == split]["dataset"].unique():
                dataset_rows = df[(df["split"] == split) & (df["dataset"] == dataset)]
                if split == "test" and pool_test_partitions:
                    first_start = dataset_rows["start"].min()
                    last_end = dataset_rows["end"].max()
                    splits[split][dataset] = [(first_start, last_end)]
                else:
                    splits[split][dataset] = []
                    for row in dataset_rows.itertuples():
                        splits[split][dataset].append((row.start, row.end))

        return cls(**splits)


# ------------------------------------------------------------------
# EmgRecording
# ------------------------------------------------------------------
class EmgRecording:
    """A read-only interface to an EMG recording of a single partition."""

    def __init__(
        self, hdf5_path: Path, start_time: float = -np.inf, end_time: float = np.inf
    ) -> None:
        self.hdf5_path = hdf5_path
        self.start_time = start_time
        self.end_time = end_time

        self._file = h5py.File(self.hdf5_path, "r")
        self.timeseries = self._file["data"]
        self.task: str = self._file["data"].attrs.get("task", "discrete_gestures")

        has_prompts = "prompts" in self._file.keys()
        self.prompts = pd.read_hdf(hdf5_path, "prompts") if has_prompts else None

        timestamps = self.timeseries["time"]
        assert (np.diff(timestamps) >= 0).all(), "Timestamps are not monotonic"
        self.start_idx, self.end_idx = timestamps.searchsorted(
            [self.start_time, self.end_time]
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._file.close()

    def __len__(self) -> int:
        return self.end_idx - self.start_idx

    def __getitem__(self, key: slice) -> np.ndarray:
        if not isinstance(key, slice):
            raise TypeError("Only slices are supported")
        start = key.start if key.start is not None else 0
        stop = key.stop if key.stop is not None else len(self)
        start += self.start_idx
        stop += self.start_idx
        return self.timeseries[start:stop]

    def get_idx_slice(
        self, start_t: float = -np.inf, end_t: float = np.inf
    ) -> tuple[Any, Any]:
        """Return (start_idx, end_idx) for the given time window."""
        assert end_t > start_t, "start_t must be less than end_t!"
        timestamps = self.timeseries["time"]
        start_idx, end_idx = timestamps.searchsorted([start_t, end_t])
        start_idx = max(start_idx, self.start_idx)
        end_idx = min(end_idx, self.end_idx)
        return start_idx, end_idx


# ------------------------------------------------------------------
# WindowedEmgDataset
# ------------------------------------------------------------------
class WindowedEmgDataset(torch.utils.data.Dataset):
    """Strided windows of EMG data from an ``EmgRecording``."""

    def __init__(
        self,
        hdf5_path: Path,
        start: float,
        end: float,
        transform: Transform,
        emg_augmentation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        window_length: int | None = 10_000,
        stride: int | None = None,
        jitter: bool = False,
    ) -> None:
        self.hdf5_path = hdf5_path
        self.start = start
        self.end = end
        self.transform = transform
        self.emg_augmentation = emg_augmentation
        self.window_length = window_length
        self.stride = stride
        self.jitter = jitter

        self.emg_recording = EmgRecording(self.hdf5_path, self.start, self.end)

        self.window_length = (
            window_length if window_length is not None else len(self.emg_recording)
        )
        self.stride = stride if stride is not None else self.window_length
        assert self.window_length > 0 and self.stride > 0

    def __len__(self) -> int:
        assert self.window_length is not None and self.stride is not None
        return int(
            max(len(self.emg_recording) - self.window_length, 0) // self.stride + 1
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        assert self.window_length is not None and self.stride is not None

        start_sample = idx * self.stride

        leftover = len(self.emg_recording) - (start_sample + self.window_length)
        if leftover < 0:
            raise IndexError(f"Index {idx} out of bounds")
        if leftover > 0 and self.jitter:
            start_sample += np.random.randint(0, min(self.stride, leftover))

        window_start = max(start_sample, 0)
        window_end = start_sample + self.window_length
        timeseries = self.emg_recording[window_start:window_end]

        datum: dict[str, torch.Tensor] = self.transform(
            timeseries, self.emg_recording.prompts
        )

        if self.emg_augmentation is not None:
            datum["emg"] = self.emg_augmentation(datum["emg"])

        is_test_mode = self.window_length == len(self.emg_recording)
        is_discrete_gestures = "discrete_gestures" in str(self.hdf5_path)

        if is_test_mode and is_discrete_gestures:
            datum["prompts"] = self.emg_recording.prompts
            datum["timestamps"] = timeseries["time"]

        return datum


# ------------------------------------------------------------------
# make_dataset
# ------------------------------------------------------------------
def make_dataset(
    data_location: str,
    partition_dict: dict[str, Partitions | None],
    transform: Transform,
    emg_augmentation: Callable[[torch.Tensor], torch.Tensor] | None,
    window_length: int | None,
    stride: int | None,
    jitter: bool,
    split_label: str | None = None,
) -> ConcatDataset:
    """Create a ``ConcatDataset`` of windowed EMG recordings."""
    datasets = []
    for dataset, partitions in tqdm(
        partition_dict.items(), desc=f"[setup] Loading datasets for split {split_label}"
    ):
        if partitions is None:
            partitions = [(-np.inf, np.inf)]

        for start, end in partitions:
            if window_length is not None:
                partition_samples = (end - start) * EMG_SAMPLE_RATE
                if partition_samples < window_length:
                    print(f"Skipping partition {dataset} {start} {end}")
                    continue

            datasets.append(
                WindowedEmgDataset(
                    get_full_dataset_path(data_location, dataset),
                    start=start,
                    end=end,
                    transform=transform,
                    window_length=window_length,
                    stride=stride,
                    jitter=jitter,
                    emg_augmentation=emg_augmentation,
                )
            )
    return ConcatDataset(datasets)


# ==================================================================
# RT-Gesture high-level helpers
# ==================================================================

@dataclass
class DataPipelineConfig:
    data_location: str
    split_csv: str
    pool_test_partitions: bool = False
    window_length: int = 16_000
    stride: int = 16_000
    batch_size: int = 64
    num_workers: int = 0
    pulse_window: tuple[float, float] = (0.08, 0.12)
    rotation_augmentation: int = 2
    include_test_loader: bool = False


def load_data_split(split_csv: str | Path, pool_test_partitions: bool = False) -> DataSplit:
    """Load train/val/test partition map from CSV."""
    return DataSplit.from_csv(str(split_csv), pool_test_partitions=pool_test_partitions)


def make_discrete_gesture_dataloaders(
    config: DataPipelineConfig,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Build DataLoaders for discrete gesture training pipeline."""
    split = load_data_split(config.split_csv, pool_test_partitions=config.pool_test_partitions)
    transform = DiscreteGesturesTransform(pulse_window=list(config.pulse_window))
    augmentation = RotationAugmentation(rotation=config.rotation_augmentation)

    train_dataset = make_dataset(
        data_location=config.data_location,
        transform=transform,
        partition_dict=split.train,
        window_length=config.window_length,
        stride=config.stride,
        jitter=True,
        emg_augmentation=augmentation,
        split_label="train",
    )
    val_dataset = make_dataset(
        data_location=config.data_location,
        transform=transform,
        partition_dict=split.val,
        window_length=config.window_length,
        stride=config.stride,
        jitter=False,
        emg_augmentation=None,
        split_label="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    test_loader: DataLoader | None = None
    if config.include_test_loader:
        test_dataset = make_dataset(
            data_location=config.data_location,
            transform=transform,
            partition_dict=split.test,
            window_length=None,
            stride=None,
            jitter=False,
            emg_augmentation=None,
            split_label="test",
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=False,
        )
    return train_loader, val_loader, test_loader


def load_full_discrete_gesture_recording(
    hdf5_path: str | Path,
    pulse_window: tuple[float, float] = (0.08, 0.12),
) -> tuple[torch.Tensor, np.ndarray, pd.DataFrame]:
    """
    Load full recording for evaluation.

    Returns
    -------
    emg_tensor:
        Shape (1, 16, T)
    times:
        Shape (T,)
    prompts:
        DataFrame with columns such as ['name', 'time']
    """
    path = Path(hdf5_path)
    with h5py.File(path, "r") as handle:
        data = handle["data"][:]
    emg = torch.from_numpy(np.stack(data["emg"], axis=0).T).float().unsqueeze(0)
    times = data["time"]
    try:
        prompts = pd.read_hdf(path, "prompts")
    except (FileNotFoundError, KeyError, OSError, ValueError):
        prompts = pd.DataFrame(columns=["name", "time"])
    return emg, times, prompts
