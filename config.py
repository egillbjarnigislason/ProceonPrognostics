"""
config.py

Per-signal-type config for two-phase transfer learning of MOMENT on KAIST
data. Mirrors VisionFramework's Config/PhaseConfig split: phase 1 trains
only the reconstruction head (frozen backbone), phase 2 unfreezes the top N
encoder blocks at a lower LR to adapt the transferred weights.

Three signal types share this same Config shape but differ in data source,
channel selection, and smoothing (see filtering.py for why smoothing is
described here as a spec rather than applied here -- it has to run on the
full signal inside dataset.py, before windowing):

    vibration_config()  .mat accel channels,        Butterworth low-pass
    current_config()    .tdms current-phase channels, Butterworth low-pass
    temp_config()       .tdms temperature channels,  moving-average

Each signal's train_<signal> entry point (see train.py) calls its matching
*_config() rather than constructing Config() directly, so the three don't
drift out of sync by hand-editing one shared object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PhaseConfig:
    epochs: int
    lr: float
    unfreeze_top_n: int  # 0 = head only (frozen backbone)


@dataclass(frozen=True)
class FilterSpec:
    """What smoothing dataset.py should apply to a signal before windowing.

    kind is "lowpass" (Butterworth, uses cutoff_hz/order) or "moving_average"
    (uses window); the fields the other kind doesn't use are left None.
    """
    kind: str
    cutoff_hz: Optional[float] = None
    order: int = 4
    window: Optional[int] = None


@dataclass
class Config:
    signal_type: str  # "vibration" | "current" | "temp" -- selects which
                       # WindowDataset class training_loop.py uses

    # DataKAIST, not Data: Data/ is a stale partial download (4 files per
    # signal) whose current/temp subfolder is even misnamed "current_temp"
    # (underscore) vs. data_loading.py's hardcoded "current,temp" (comma) --
    # that combination would silently break current/temp loading. DataKAIST/
    # is the complete, correctly-named 41-condition download (matches the
    # comma folder data_loading.py expects) and is what NB_DataExpl.ipynb
    # already uses.
    data_dir: Path = Path("DataKAIST")

    # Vibration only: which of VIBRATION_CHANNELS to use. Current/temp
    # datasets instead take every channel data_loading.py reports as
    # populated for a given file (channel names/count there are dynamic,
    # not a fixed list -- see dataset.py).
    channels: List[str] = field(default_factory=list)

    # File-level split (never split within a file) so overlapping train/val
    # windows never share near-duplicate content.
    train_stems: List[str] = field(default_factory=lambda: ["0Nm_Normal", "2Nm_Normal"])
    val_stems: List[str] = field(default_factory=lambda: ["4Nm_Normal"])

    window: int = 512
    train_stride: int = 128  # overlapping: more windows for fine-tuning
    val_stride: int = 512    # non-overlapping: clean, non-redundant val signal
    mask_ratio: float = 0.3

    filter_spec: Optional[FilterSpec] = None

    batch_size: int = 32
    device: str = "cpu"

    phases: List[PhaseConfig] = field(default_factory=list)

    checkpoint_path: Path = Path("checkpoints/moment.pt")


def vibration_config() -> Config:
    """Repeat of the original run_29433679 vibration fine-tune, with the
    Butterworth denoise from NB_DataExpl.ipynb's PSD check folded into the
    training data, and phase 2 extended 5 -> 16 epochs.
    """
    return Config(
        signal_type="vibration",
        channels=["accel_x_A", "accel_y_A"],
        filter_spec=FilterSpec(kind="lowpass", cutoff_hz=10_000),
        phases=[
            PhaseConfig(epochs=3, lr=1e-3, unfreeze_top_n=0),
            PhaseConfig(epochs=16, lr=1e-5, unfreeze_top_n=4),
        ],
        checkpoint_path=Path("checkpoints/vibration/moment.pt"),
    )


def current_config() -> Config:
    """cutoff_hz=800: the PSD-based recommendation (removes the 3k/6k/9k/12k
    harmonic clusters and leaves the fundamental) -- NOT the 12000 Hz value
    that was sitting in the notebook, which sat right next to this signal's
    ~12.8kHz Nyquist and removed almost nothing.
    """
    return Config(
        signal_type="current",
        filter_spec=FilterSpec(kind="lowpass", cutoff_hz=800),
        phases=[
            PhaseConfig(epochs=3, lr=1e-3, unfreeze_top_n=0),
            PhaseConfig(epochs=10, lr=1e-5, unfreeze_top_n=4),
        ],
        checkpoint_path=Path("checkpoints/current/moment.pt"),
    )


def temp_config() -> Config:
    """Moving-average smoothing, window=4000 samples, matching the value
    already used for current in NB_DataExpl.ipynb.

    Note: at a ~25.6kHz sample rate, a 512-sample window (config.window,
    same as vibration/current) spans only ~20ms -- far shorter than any real
    thermal dynamics. Windows will mostly look like near-flat segments of a
    slowly drifting signal, which the model may just learn to reconstruct as
    close to constant. That's an expected property of this signal, not a
    setup bug, but worth watching in the val-loss curve.
    """
    return Config(
        signal_type="temp",
        filter_spec=FilterSpec(kind="moving_average", window=4000),
        phases=[
            PhaseConfig(epochs=3, lr=1e-3, unfreeze_top_n=0),
            PhaseConfig(epochs=10, lr=1e-5, unfreeze_top_n=4),
        ],
        checkpoint_path=Path("checkpoints/temp/moment.pt"),
    )
