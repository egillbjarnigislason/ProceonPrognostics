"""
data_loading.py

Minimal loaders for the KAIST rotating-machine dataset (vibration .mat +
current/temperature .tdms). Structure below was confirmed by directly
inspecting sample files, not assumed from documentation.

File pairing: every condition has one file in "Data/vibration/<stem>.mat"
and one in "Data/current,temp/<stem>.tdms" sharing the same <stem>, e.g.
"2Nm_BPFO_03".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import scipy.io as sio
from nptdms import TdmsFile


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# Matches e.g. "2Nm_BPFO_03", "0Nm_Normal", "4Nm_Unbalance_0583mg",
# "2Nm_Unbalalnce_1169mg" (dataset typo in the 2Nm files — tolerated below).
_FILENAME_RE = re.compile(
    r"^(?P<load_nm>\d+)Nm_(?P<condition>Normal|BPFI|BPFO|Misalign|Unbalance|Unbalalnce)"
    r"(?:_(?P<severity>\w+))?$"
)


@dataclass(frozen=True)
class ConditionInfo:
    stem: str                  # e.g. "2Nm_BPFO_03" — used to find both files
    load_nm: int                # 0, 2, or 4
    condition: str               # "Normal" | "BPFI" | "BPFO" | "Misalign" | "Unbalance"
    severity: Optional[str]      # e.g. "03", "05", "0583mg"; None for Normal
    is_normal: bool


def parse_condition_filename(stem: str) -> ConditionInfo:
    """Parse a KAIST filename stem (no extension) into its condition fields.

    Tolerates the dataset's "Unbalalnce" typo (present in the 2Nm files) and
    normalizes it to "Unbalance".
    """
    m = _FILENAME_RE.match(stem)
    if not m:
        raise ValueError(f"Filename stem does not match expected pattern: {stem!r}")
    condition = m.group("condition")
    if condition == "Unbalalnce":
        condition = "Unbalance"
    return ConditionInfo(
        stem=stem,
        load_nm=int(m.group("load_nm")),
        condition=condition,
        severity=m.group("severity"),
        is_normal=(condition == "Normal"),
    )


# ---------------------------------------------------------------------------
# Vibration (.mat)
# ---------------------------------------------------------------------------

# NB: the file itself only labels these channels "Point1".."Point4" — it does
# not state which is x/y or housing A/B anywhere in the .mat metadata. This
# ordering follows the dataset's published description (x_A, y_A, x_B, y_B)
# but is NOT independently confirmed from the file — treat as provisional
# until cross-checked against the dataset's own README/documentation.
VIBRATION_CHANNELS = ["accel_x_A", "accel_y_A", "accel_x_B", "accel_y_B"]


@dataclass
class VibrationSignal:
    values: np.ndarray       # (n_samples, 4) float64, units g
    sample_rate_hz: float
    channel_names: List[str]


def load_vibration_mat(path) -> VibrationSignal:
    """Load one Test.Lab-exported vibration .mat file.

    Confirmed structure:
        Signal.x_values.increment  -> 1 / sample_rate  (seconds)
        Signal.y_values.values     -> (n_samples, 4) float64, units 'g'
    """
    mat = sio.loadmat(str(path), struct_as_record=False, squeeze_me=False)
    signal = mat["Signal"][0, 0]

    x = signal.x_values[0, 0]
    increment = float(x.increment[0, 0])
    sample_rate_hz = 1.0 / increment

    y = signal.y_values[0, 0]
    values = np.asarray(y.values, dtype=np.float64)  # (n_samples, 4)

    return VibrationSignal(
        values=values,
        sample_rate_hz=sample_rate_hz,
        channel_names=list(VIBRATION_CHANNELS),
    )


# ---------------------------------------------------------------------------
# Current + temperature (.tdms)
# ---------------------------------------------------------------------------

@dataclass
class CurrentTempSignal:
    temperature: np.ndarray          # (n_samples, n_temp_channels) degrees C
    temperature_channels: List[str]
    current: np.ndarray              # (n_samples, n_current_channels) amps
    current_channels: List[str]      # only channels that actually had data
    sample_rate_hz: float


def load_current_temp_tdms(path) -> CurrentTempSignal:
    """Load one FlexLogger .tdms file (temperature + current channels).

    Not every file has all 3 current phases populated — confirmed: the
    2Nm_BPFO_03 file only has channel "ai0" with real samples; "ai2"/"ai3"
    exist in the file but have zero length. This function drops any channel
    with zero samples and reports which ones survived via
    `current_channels`, rather than assuming a fixed channel count.
    """
    with TdmsFile.open(str(path)) as tdms:
        log = tdms["Log"]

        temp_cols, temp_names = [], []
        curr_cols, curr_names = [], []
        sample_rate_hz = None

        for ch in log.channels():
            props = ch.properties
            ch_type = props.get("DAC~Channel~Type")
            n = len(ch)
            if n == 0:
                continue  # unpopulated channel for this condition -- skip

            if sample_rate_hz is None:
                sample_rate_hz = 1.0 / float(props["wf_increment"])

            data = ch[:].astype(np.float64)
            if ch_type == "Temperature":
                temp_cols.append(data)
                temp_names.append(ch.name)
            elif ch_type == "Current":
                curr_cols.append(data)
                curr_names.append(ch.name)

    if sample_rate_hz is None:
        raise ValueError(f"No populated channels found in {path}")

    # Channel lengths can differ by a handful of samples; trim to shortest
    # before stacking into one array.
    def stack(cols):
        if not cols:
            return np.zeros((0, 0), dtype=np.float64)
        min_len = min(len(c) for c in cols)
        return np.stack([c[:min_len] for c in cols], axis=1)

    return CurrentTempSignal(
        temperature=stack(temp_cols),
        temperature_channels=temp_names,
        current=stack(curr_cols),
        current_channels=curr_names,
        sample_rate_hz=sample_rate_hz,
    )


# ---------------------------------------------------------------------------
# Pairing vibration + current/temp for one condition
# ---------------------------------------------------------------------------

@dataclass
class ConditionRecord:
    info: ConditionInfo
    vibration: VibrationSignal
    current_temp: CurrentTempSignal


def load_condition(data_dir, stem: str) -> ConditionRecord:
    """Load both files for one condition stem, e.g. "2Nm_BPFO_03"."""
    data_dir = Path(data_dir)
    info = parse_condition_filename(stem)
    vibration = load_vibration_mat(data_dir / "vibration" / f"{stem}.mat")
    current_temp = load_current_temp_tdms(data_dir / "current,temp" / f"{stem}.tdms")
    return ConditionRecord(info=info, vibration=vibration, current_temp=current_temp)


def list_condition_stems(data_dir) -> List[str]:
    """Every condition stem that has BOTH a .mat and a .tdms file on disk."""
    data_dir = Path(data_dir)
    mat_stems = {p.stem for p in (data_dir / "vibration").glob("*.mat")}
    tdms_stems = {p.stem for p in (data_dir / "current,temp").glob("*.tdms")}
    missing_tdms = mat_stems - tdms_stems
    missing_mat = tdms_stems - mat_stems
    if missing_tdms:
        print(f"[data] WARNING: no .tdms match for: {sorted(missing_tdms)}")
    if missing_mat:
        print(f"[data] WARNING: no .mat match for: {sorted(missing_mat)}")
    return sorted(mat_stems & tdms_stems)
