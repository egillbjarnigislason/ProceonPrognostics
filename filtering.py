"""
filtering.py

Smoothing applied to a full continuous recording BEFORE it is windowed --
never after. A zero-phase filter (filtfilt) pads/reflects at the edges of
whatever array you hand it; if that array is already a slice of a longer
recording, the slice boundary gets treated as if the recording genuinely
ended there, producing a spurious transient at the cut point. This bit us
once already, in NB_DataExpl.ipynb's plot_mat_data (subset sliced before
filtering) -- these functions exist so the training pipeline never repeats
that mistake: dataset.py always calls these on an entire loaded file, then
windows the *result*.

Both functions operate on plain (n_samples, n_channels) numpy arrays so
dataset.py doesn't need to round-trip through pandas.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt


def lowpass_filter(values: np.ndarray, cutoff_hz: float, sample_rate_hz: float,
                    order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass, applied independently per channel.

    cutoff_hz is the -3dB rolloff point; sample_rate_hz must be the true
    sampling rate of `values` (e.g. sig.sample_rate_hz / ct.sample_rate_hz
    from data_loading.py), not a guessed number.
    """
    nyquist = sample_rate_hz / 2
    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    return np.stack(
        [filtfilt(b, a, values[:, c]) for c in range(values.shape[1])], axis=1
    )


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving-average smoothing, applied independently per channel.
    window is in samples. Edges use "nearest" padding (scipy's
    uniform_filter1d) rather than pandas' shrinking-window behavior at the
    boundary -- fine here since these are denoising full recordings, not
    doing edge-sensitive analysis.
    """
    return np.stack(
        [uniform_filter1d(values[:, c], size=window, mode="nearest")
         for c in range(values.shape[1])],
        axis=1,
    )
