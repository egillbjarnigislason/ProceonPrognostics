"""
evaluate.py

Rolling-window reconstruction over a full time series -- the same unmasked,
one-window-at-a-time computation the model would run on live data (see
training_loop.py's unmasked validation pass), applied here across an entire
signal at once so raw vs. fine-tuned reconstructions can be compared against
the actual data.
"""

from __future__ import annotations

import time

import numpy as np
import torch


def rolling_reconstruct(model, values: np.ndarray, window: int = 512, stride: int = 512,
                         batch_size: int = 32, device: str = "cpu", log_every: int = 5):
    """Slide a `window`-length window across `values` (n_samples, n_channels)
    with the given `stride`, reconstruct each window with `model` -- no
    masking, exactly like validation / real deployment -- and reassemble the
    per-window reconstructions back into one continuous series covering the
    same span as the input (overlapping windows are averaged).

    Use `stride <= window` for full coverage; a larger stride would leave
    gaps, which are simply dropped from the output rather than filled in.

    Prints batch progress/ETA every `log_every` batches (set to 0 to disable),
    same pattern as training_loop.py's _run_epoch -- useful since a full-signal
    pass on CPU can take minutes and would otherwise look hung.

    Returns (sample_idx, actual, reconstructed):
      sample_idx    -- (n_covered,) indices into `values`
      actual        -- (n_covered, n_channels) original values at those indices
      reconstructed -- (n_covered, n_channels) model reconstruction, averaged
                       over any overlapping windows
    """
    model.eval()
    n_samples, n_channels = values.shape
    starts = list(range(0, n_samples - window + 1, stride))
    n_batches = max(1, -(-len(starts) // batch_size))  # ceil div

    recon_sum = np.zeros((n_samples, n_channels), dtype=np.float64)
    count = np.zeros(n_samples, dtype=np.int32)

    print(f"[rolling_reconstruct] {len(starts)} windows over {n_batches} batches "
          f"(window={window} stride={stride} batch_size={batch_size} device={device})", flush=True)
    t0 = time.time()

    with torch.no_grad():
        for b, i in enumerate(range(0, len(starts), batch_size), start=1):
            batch_starts = starts[i:i + batch_size]
            batch_np = np.stack([values[s:s + window].T for s in batch_starts], axis=0)
            batch = torch.from_numpy(batch_np.astype(np.float32)).to(device)  # (b, n_channels, window)

            input_mask = torch.ones(batch.shape[0], batch.shape[-1], device=device)
            mask = torch.ones_like(input_mask)  # no masking -- real-use reconstruction
            output = model(x_enc=batch, input_mask=input_mask, mask=mask)
            recon = output.reconstruction.cpu().numpy()  # (b, n_channels, window)

            for j, s in enumerate(batch_starts):
                recon_sum[s:s + window] += recon[j].T  # (window, n_channels)
                count[s:s + window] += 1

            if log_every and (b % log_every == 0 or b == n_batches):
                elapsed = time.time() - t0
                eta = elapsed / b * (n_batches - b)
                print(f"    [rolling_reconstruct] batch {b}/{n_batches} "
                      f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    covered = count > 0
    sample_idx = np.nonzero(covered)[0]
    actual = values[sample_idx]
    reconstructed = (recon_sum[sample_idx] / count[sample_idx, None]).astype(np.float32)
    return sample_idx, actual, reconstructed
