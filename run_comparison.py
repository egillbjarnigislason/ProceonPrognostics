"""
run_comparison.py

Hardcoded raw-vs-fine-tuned MOMENT reconstruction comparison on one file
(4Nm_Normal), moved out of FineTunedExplore.ipynb so the slow full-signal
rolling_reconstruct pass can run headless from the terminal instead of
tying up a notebook kernel. Writes sample_idx/actual/raw/fine-tuned to a
CSV for plotting later in a notebook.

    python3 run_comparison.py
"""

from pathlib import Path

import numpy as np
import torch

from config import vibration_config
from data_loading import VIBRATION_CHANNELS, load_vibration_mat
from evaluate import rolling_reconstruct
from model import load_moment

STEM = "4Nm_Unbalance_3318mg"
FINETUNED_CKPT = Path("outputs/run_29433679/vibration_xy_moment.pt")
OUT_CSV = Path("outputs/comparison") / f"{STEM}_reconstruction.csv"


def main() -> None:
    # vibration_config(), not Config() directly -- config.py now has one
    # Config shape shared by three signal-specific factories (see below).
    # NOTE: this comparison still loads the file's raw, unfiltered values --
    # FINETUNED_CKPT (run_29433679) was itself trained before the BW-filter
    # setup existed, so filtering here would just make raw vs. fine-tuned
    # harder to compare apples to apples. Once a filtered vibration
    # checkpoint exists (config.py:vibration_config), point FINETUNED_CKPT
    # at it and apply the same filter to `values` below for a fair
    # comparison.
    cfg = vibration_config()
    device = torch.device(cfg.device)

    print(f"[load] {STEM}", flush=True)
    sig = load_vibration_mat(cfg.data_dir / "vibration" / f"{STEM}.mat")
    chan_idx = [VIBRATION_CHANNELS.index(c) for c in cfg.channels]
    values = sig.values[:, chan_idx]  # (n_samples, len(cfg.channels))
    print(f"[load] values shape={values.shape}", flush=True)

    print("[model] loading raw MOMENT", flush=True)
    raw_model = load_moment(cfg.mask_ratio).to(device)

    print(f"[model] loading fine-tuned checkpoint: {FINETUNED_CKPT}", flush=True)
    finetuned_model = load_moment(cfg.mask_ratio).to(device)
    finetuned_model.load_state_dict(torch.load(FINETUNED_CKPT, map_location=device))

    print("[infer] raw model", flush=True)
    idx, actual, recon_raw = rolling_reconstruct(
        raw_model, values, window=cfg.window, stride=cfg.window, device=cfg.device
    )
    print("[infer] fine-tuned model", flush=True)
    _, _, recon_ft = rolling_reconstruct(
        finetuned_model, values, window=cfg.window, stride=cfg.window, device=cfg.device
    )

    mse_raw = ((recon_raw - actual) ** 2).mean(axis=0)
    mse_ft = ((recon_ft - actual) ** 2).mean(axis=0)
    print(f"[result] per-channel MSE  raw={mse_raw}  finetuned={mse_ft}", flush=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    columns = {"sample_idx": idx}
    for c, name in enumerate(cfg.channels):
        columns[f"actual_{name}"] = actual[:, c]
        columns[f"raw_{name}"] = recon_raw[:, c]
        columns[f"finetuned_{name}"] = recon_ft[:, c]

    header = ",".join(columns.keys())
    data = np.column_stack(list(columns.values()))
    np.savetxt(OUT_CSV, data, delimiter=",", header=header, comments="", fmt="%.6g")
    print(f"[done] wrote {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
