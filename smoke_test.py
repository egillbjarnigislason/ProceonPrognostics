"""
smoke_test.py

    python3 smoke_test.py --signal {vibration,current,temp}

Quick local sanity check that a signal's training pipeline runs end to end --
data loads, channels line up, MOMENT loads, freeze/unfreeze works, a
forward/backward pass completes, a checkpoint gets written -- without
sitting through a real multi-epoch run. NOT a real training run: forces a
single 1-epoch head-only phase and stretches the window stride so only a
handful of windows get built (a few dozen instead of tens of thousands), so
it finishes in well under a couple of minutes even on a CPU-only laptop.

Writes to checkpoints/<signal>_smoke/ -- a separate directory from the real
checkpoints/<signal>/ used by train.py, so this can never overwrite a real
run's checkpoint or history.csv.

Passing does NOT validate the HPC environment itself (module versions,
network access from a compute node, etc.) -- only that the code and config
are sound. Still run the real hpc/train_<signal>.sh job to confirm that
separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import PhaseConfig, current_config, temp_config, vibration_config
from training_loop import train

_CONFIGS = {
    "vibration": vibration_config,
    "current": current_config,
    "temp": temp_config,
}

_SMOKE_STRIDE = 100_000  # samples -- large enough that only ~dozens of windows get built


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", choices=sorted(_CONFIGS), required=True)
    args = parser.parse_args()

    cfg = _CONFIGS[args.signal]()
    cfg.train_stride = _SMOKE_STRIDE
    cfg.val_stride = _SMOKE_STRIDE
    cfg.phases = [PhaseConfig(epochs=1, lr=cfg.phases[0].lr, unfreeze_top_n=0)]
    cfg.checkpoint_path = Path(f"checkpoints/{args.signal}_smoke/moment.pt")

    print(f"[smoke_test] signal={args.signal} data_dir={cfg.data_dir} "
          f"train_stems={cfg.train_stems} val_stems={cfg.val_stems} "
          f"filter={cfg.filter_spec}", flush=True)

    result = train(cfg)
    print(f"[smoke_test] OK -- signal={args.signal} best_val_loss={result.best_val_loss:.6f} "
          f"ckpt={result.best_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
