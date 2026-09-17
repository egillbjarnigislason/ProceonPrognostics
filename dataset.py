"""
dataset.py

Windows the vibration channels from Normal-condition KAIST files into
fixed-length (n_channels, window) tensors for MOMENT. Split is done at the
file level (train_stems vs val_stems) rather than by re-splitting windows
after the fact, so overlapping train/val windows never share near-duplicate
content.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from data_loading import VIBRATION_CHANNELS, load_vibration_mat


def _windows(values: np.ndarray, window: int, stride: int) -> np.ndarray:
    """(n_samples, n_channels) -> (n_windows, n_channels, window)."""
    n_samples = values.shape[0]
    starts = range(0, n_samples - window + 1, stride)
    return np.stack([values[s:s + window].T for s in starts], axis=0).astype(np.float32)


class VibrationWindowDataset(Dataset):
    def __init__(self, data_dir: Path, stems: List[str], channels: List[str],
                 window: int, stride: int):
        chan_idx = [VIBRATION_CHANNELS.index(c) for c in channels]
        chunks = []
        for stem in stems:
            sig = load_vibration_mat(Path(data_dir) / "vibration" / f"{stem}.mat")
            values = sig.values[:, chan_idx]  # (n_samples, len(channels))
            chunks.append(_windows(values, window, stride))
        self.data = np.concatenate(chunks, axis=0)  # (n_windows, n_channels, window)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.data[idx])  # (n_channels, window)
