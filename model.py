"""
model.py

Thin wrapper around MOMENTPipeline for reconstruction-based fine-tuning.
Mirrors VisionFramework's backbone freeze / set_backbone_trainable pattern:
the pretrained encoder starts fully frozen (only the reconstruction head is
trainable), and unfreeze_top_blocks() opens up the top N encoder blocks for
phase-2 fine-tuning.

mask_ratio is passed in at load time: MOMENT keeps its own mask_generator
internally and auto-generates a fresh random reconstruction mask on every
forward call, so training_loop.py never has to touch masking directly.
"""

from __future__ import annotations

import re

from momentfm import MOMENTPipeline

_MODEL_NAME = "AutonLab/MOMENT-1-large"
_BLOCK_RE = re.compile(r"encoder\.block\.(\d+)\.")


def load_moment(mask_ratio: float) -> MOMENTPipeline:
    model = MOMENTPipeline.from_pretrained(
        _MODEL_NAME,
        model_kwargs={"task_name": "reconstruction", "mask_ratio": mask_ratio},
    )
    model.init()
    freeze_backbone(model)
    return model


def freeze_backbone(model) -> None:
    """Freeze everything except the reconstruction head (phase-1 state)."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("head.")


def unfreeze_top_blocks(model, n: int) -> None:
    """On top of the frozen backbone, unfreeze the last n encoder blocks."""
    if n <= 0:
        return
    block_ids = sorted({
        int(m.group(1)) for name, _ in model.named_parameters()
        if (m := _BLOCK_RE.search(name))
    })
    top_ids = set(block_ids[-n:])
    for name, param in model.named_parameters():
        m = _BLOCK_RE.search(name)
        if m and int(m.group(1)) in top_ids:
            param.requires_grad = True


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]
