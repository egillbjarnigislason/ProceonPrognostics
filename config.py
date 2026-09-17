"""
config.py

Minimal config for two-phase transfer learning of MOMENT on KAIST vibration
data. Mirrors VisionFramework's Config/PhaseConfig split: phase 1 trains only
the reconstruction head (frozen backbone), phase 2 unfreezes the top N
encoder blocks at a lower LR to adapt the transferred weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class PhaseConfig:
    epochs: int
    lr: float
    unfreeze_top_n: int  # 0 = head only (frozen backbone)


@dataclass
class Config:
    data_dir: Path = Path("Data")
    channels: List[str] = field(default_factory=lambda: ["accel_x_A", "accel_y_A"])

    # File-level split (never split within a file) to avoid leaking
    # near-duplicate overlapping windows across train/val.
    train_stems: List[str] = field(default_factory=lambda: ["0Nm_Normal", "2Nm_Normal"])
    val_stems: List[str] = field(default_factory=lambda: ["4Nm_Normal"])

    window: int = 512
    train_stride: int = 128  # overlapping: more windows for fine-tuning
    val_stride: int = 512    # non-overlapping: clean, non-redundant val signal
    mask_ratio: float = 0.3

    batch_size: int = 32
    device: str = "cpu"

    phases: List[PhaseConfig] = field(default_factory=lambda: [
        PhaseConfig(epochs=3, lr=1e-3, unfreeze_top_n=0),  # phase1: warmup, head only
        PhaseConfig(epochs=5, lr=1e-5, unfreeze_top_n=4),  # phase2: finetune top 4 blocks
    ])

    checkpoint_path: Path = Path("checkpoints/vibration_xy_moment.pt")
