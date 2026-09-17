"""
train.py

    python train.py

Runs the two-phase MOMENT fine-tune on KAIST vibration (accel_x_A, accel_y_A)
Normal-condition data, per config.py.
"""

from config import Config
from training_loop import train

if __name__ == "__main__":
    result = train(Config())
    print(f"done. best_val_loss={result.best_val_loss:.6f} ckpt={result.best_checkpoint}")
