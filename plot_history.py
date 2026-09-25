"""
plot_history.py

    python plot_history.py [path/to/history.csv]

Plots train/val loss over training from the CSV training_loop.py writes
(default: checkpoints/history.csv, matching config.py's checkpoint_path).
Can be run after training finishes, or at any point during a long run --
the CSV is flushed after every epoch, so it always reflects progress so far.
Saves a PNG next to the CSV.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return [
            {
                "global_epoch": int(row["global_epoch"]),
                "phase": int(row["phase"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
            }
            for row in csv.DictReader(f)
        ]


def plot(history: list[dict], out_path: Path) -> None:
    epochs = [r["global_epoch"] for r in history]
    plt.plot(epochs, [r["train_loss"] for r in history], label="train")
    plt.plot(epochs, [r["val_loss"] for r in history], label="val (unmasked)")

    # vertical line at each phase boundary
    phases = [r["phase"] for r in history]
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            plt.axvline(epochs[i] - 1.0, color="gray", linestyle="--", linewidth=0.8)

    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("MOMENT fine-tune: train/val loss")
    plt.legend()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints/history.csv")
    plot(load_history(csv_path), csv_path.with_suffix(".png"))
