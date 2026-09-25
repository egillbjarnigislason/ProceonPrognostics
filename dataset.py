"""
dataset.py

Windows a continuous signal into fixed-length (n_channels, window) tensors
for MOMENT, applying whatever config.FilterSpec says first -- filtering
always runs on the entire loaded file, then _windows() slices the *result*,
never the other way around (see filtering.py for why).

Split is done at the file level (train_stems vs val_stems) rather than by
re-splitting windows after the fact, so overlapping train/val windows never
share near-duplicate content. One dataset class per signal type, since each
loads a different file kind and selects channels differently:

    VibrationWindowDataset    .mat, fixed channel list (config.channels)
    CurrentWindowDataset      .tdms, every populated current-phase channel
    TemperatureWindowDataset  .tdms, every populated temperature channel

Current/temp channel names and counts come from the file itself (see
data_loading.load_current_temp_tdms's docstring -- not every file has all 3
phases populated), so those two classes check that every stem in a given
split reports the same channel count and refuse to mix stems that don't --
windows from different stems have to share one channel axis to be
concatenated into a single dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from config import FilterSpec
from data_loading import (
    VIBRATION_CHANNELS,
    load_current_temp_tdms,
    load_vibration_mat,
)
from filtering import lowpass_filter, moving_average


def _windows(values: np.ndarray, window: int, stride: int) -> np.ndarray:
    """(n_samples, n_channels) -> (n_windows, n_channels, window)."""
    n_samples = values.shape[0]
    starts = range(0, n_samples - window + 1, stride)
    return np.stack([values[s:s + window].T for s in starts], axis=0).astype(np.float32)


def _apply_filter(values: np.ndarray, sample_rate_hz: float,
                   filter_spec: Optional[FilterSpec]) -> np.ndarray:
    if filter_spec is None:
        return values
    if filter_spec.kind == "lowpass":
        return lowpass_filter(values, filter_spec.cutoff_hz, sample_rate_hz, filter_spec.order)
    if filter_spec.kind == "moving_average":
        return moving_average(values, filter_spec.window)
    raise ValueError(f"Unknown filter kind: {filter_spec.kind!r}")


class VibrationWindowDataset(Dataset):
    def __init__(self, data_dir: Path, stems: List[str], channels: List[str],
                 window: int, stride: int, filter_spec: Optional[FilterSpec] = None):
        chan_idx = [VIBRATION_CHANNELS.index(c) for c in channels]
        chunks = []
        for stem in stems:
            sig = load_vibration_mat(Path(data_dir) / "vibration" / f"{stem}.mat")
            values = sig.values[:, chan_idx]  # (n_samples, len(channels))
            values = _apply_filter(values, sig.sample_rate_hz, filter_spec)
            chunks.append(_windows(values, window, stride))
        self.data = np.concatenate(chunks, axis=0)  # (n_windows, n_channels, window)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.data[idx])  # (n_channels, window)


class CurrentWindowDataset(Dataset):
    def __init__(self, data_dir: Path, stems: List[str], window: int, stride: int,
                 filter_spec: Optional[FilterSpec] = None):
        chunks = []
        expected_channels: Optional[List[str]] = None
        for stem in stems:
            ct = load_current_temp_tdms(Path(data_dir) / "current,temp" / f"{stem}.tdms")
            if expected_channels is None:
                expected_channels = ct.current_channels
            elif ct.current_channels != expected_channels:
                raise ValueError(
                    f"{stem}: current channels {ct.current_channels} don't match "
                    f"{expected_channels} from an earlier stem in this split -- "
                    f"can't concatenate windows across a mismatched channel set"
                )
            values = _apply_filter(ct.current, ct.sample_rate_hz, filter_spec)
            chunks.append(_windows(values, window, stride))
        self.data = np.concatenate(chunks, axis=0)
        self.channel_names = expected_channels or []

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.data[idx])


class TemperatureWindowDataset(Dataset):
    def __init__(self, data_dir: Path, stems: List[str], window: int, stride: int,
                 filter_spec: Optional[FilterSpec] = None):
        chunks = []
        expected_channels: Optional[List[str]] = None
        for stem in stems:
            ct = load_current_temp_tdms(Path(data_dir) / "current,temp" / f"{stem}.tdms")
            if expected_channels is None:
                expected_channels = ct.temperature_channels
            elif ct.temperature_channels != expected_channels:
                raise ValueError(
                    f"{stem}: temperature channels {ct.temperature_channels} don't "
                    f"match {expected_channels} from an earlier stem in this split -- "
                    f"can't concatenate windows across a mismatched channel set"
                )
            values = _apply_filter(ct.temperature, ct.sample_rate_hz, filter_spec)
            chunks.append(_windows(values, window, stride))
        self.data = np.concatenate(chunks, axis=0)
        self.channel_names = expected_channels or []

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.data[idx])
