"""
train.py

    python3 train.py --signal {vibration,current,temp}

Runs the two-phase MOMENT fine-tune for one signal type, using that signal's
config.py factory (vibration_config / current_config / temp_config). Each
HPC job script under hpc/ calls this with a different --signal so the three
fine-tunes run as separate jobs with separate checkpoints/<signal>/ outputs.
"""

import argparse

from config import current_config, temp_config, vibration_config
from training_loop import train

_CONFIGS = {
    "vibration": vibration_config,
    "current": current_config,
    "temp": temp_config,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", choices=sorted(_CONFIGS), required=True)
    args = parser.parse_args()

    cfg = _CONFIGS[args.signal]()
    result = train(cfg)
    print(f"done. signal={args.signal} best_val_loss={result.best_val_loss:.6f} "
          f"ckpt={result.best_checkpoint}")


if __name__ == "__main__":
    main()
