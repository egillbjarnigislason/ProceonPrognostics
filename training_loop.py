"""
training_loop.py

Two-phase fine-tuning loop, mirroring VisionFramework's phase1_warmup /
phase2_finetune pattern: each phase trains for its own epoch count and LR,
and the best checkpoint (lowest val loss) is tracked globally across phases
rather than reset at each phase boundary.

One loop, three signal types: cfg.signal_type selects which WindowDataset
class to build (see dataset.py) -- vibration/current/temp otherwise share
every other step (masking, optimizer, checkpointing, history logging).

Per-epoch train/val loss is also logged to a CSV next to the checkpoint
(flushed after every epoch, so it survives an interrupted run) -- plot it
with plot_history.py.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from dataset import CurrentWindowDataset, TemperatureWindowDataset, VibrationWindowDataset
from model import load_moment, trainable_params, unfreeze_top_blocks

_DATASET_CLASSES = {
    "vibration": VibrationWindowDataset,
    "current": CurrentWindowDataset,
    "temp": TemperatureWindowDataset,
}


@dataclass
class TrainResult:
    best_val_loss: float
    best_checkpoint: Path
    history_path: Path
    history: List[dict] = field(default_factory=list)


def _build_dataset(cfg: Config, stems: List[str], stride: int):
    try:
        cls = _DATASET_CLASSES[cfg.signal_type]
    except KeyError:
        raise ValueError(
            f"Unknown signal_type {cfg.signal_type!r}; expected one of "
            f"{sorted(_DATASET_CLASSES)}"
        )
    if cfg.signal_type == "vibration":
        return cls(cfg.data_dir, stems, cfg.channels, cfg.window, stride, cfg.filter_spec)
    return cls(cfg.data_dir, stems, cfg.window, stride, cfg.filter_spec)


def _run_epoch(model, loader, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, total_n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)  # (B, n_channels, window)
        input_mask = torch.ones(batch.shape[0], batch.shape[-1], device=device)
        # Training: leave mask=None so MOMENT auto-generates a fresh random
        # reconstruction mask each call (the masked-reconstruction training
        # objective). Validation: force an all-ones mask (nothing hidden),
        # so val loss is the genuine full-window reconstruction error --
        # the same computation used for real anomaly scoring -- instead of
        # a masked-fill-in error that would also differ randomly between
        # identical models/runs, since a random mask is not reproducible.
        mask = None if training else torch.ones_like(input_mask)
        with torch.set_grad_enabled(training):
            output = model(x_enc=batch, input_mask=input_mask, mask=mask)
            loss = nn.functional.mse_loss(output.reconstruction, batch)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * batch.shape[0]
        total_n += batch.shape[0]
    return total_loss / total_n


def train(cfg: Config) -> TrainResult:
    device = torch.device(cfg.device)
    model = load_moment(cfg.mask_ratio).to(device)

    train_ds = _build_dataset(cfg, cfg.train_stems, cfg.train_stride)
    val_ds = _build_dataset(cfg, cfg.val_stems, cfg.val_stride)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    print(f"[{cfg.signal_type}][data] train windows={len(train_ds)} val windows={len(val_ds)}")

    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history: List[dict] = []

    history_path = cfg.checkpoint_path.parent / "history.csv"
    with open(history_path, "w", newline="") as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=["global_epoch", "phase", "epoch_in_phase", "train_loss", "val_loss"],
        )
        writer.writeheader()

        global_epoch = 0
        for i, phase in enumerate(cfg.phases, start=1):
            unfreeze_top_blocks(model, phase.unfreeze_top_n)
            optimizer = torch.optim.Adam(trainable_params(model), lr=phase.lr)
            n_trainable = sum(p.numel() for p in trainable_params(model))
            print(f"[{cfg.signal_type}][phase {i}] epochs={phase.epochs} lr={phase.lr} "
                  f"unfreeze_top_n={phase.unfreeze_top_n} trainable_params={n_trainable}")

            for epoch in range(1, phase.epochs + 1):
                global_epoch += 1
                train_loss = _run_epoch(model, train_loader, device, optimizer)
                val_loss = _run_epoch(model, val_loader, device, optimizer=None)
                print(f"[{cfg.signal_type}][phase {i}][epoch {epoch}] "
                      f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

                row = {"global_epoch": global_epoch, "phase": i, "epoch_in_phase": epoch,
                       "train_loss": train_loss, "val_loss": val_loss}
                history.append(row)
                writer.writerow(row)
                history_file.flush()  # survive a crash/interrupt mid-run

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), cfg.checkpoint_path)

    return TrainResult(best_val_loss=best_val_loss, best_checkpoint=cfg.checkpoint_path,
                        history_path=history_path, history=history)
