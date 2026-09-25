# HPC job scripts (DTU `dcc` cluster, LSF/`bsub`)

Three independent fine-tuning runs, one per signal type. Each script just
loads the venv and calls `train.py --signal <name>` (see `../config.py` for
what each signal's schedule/filter actually is):

| script | signal | filter | phases (epochs) |
|---|---|---|---|
| `train_vibration.sh` | vibration | Butterworth low-pass, 10 kHz | 3 + 16 |
| `train_current.sh`   | current   | Butterworth low-pass, 800 Hz | 3 + 10 |
| `train_temp.sh`      | temp      | moving average, window=4000  | 3 + 10 |

## Submitting

From the repo root on the cluster (a `.venv` with `requirements.txt`
installed must already exist there):

```sh
bsub < hpc/train_vibration.sh
bsub < hpc/train_current.sh
bsub < hpc/train_temp.sh
```

Each is independent -- submit one, all three, or re-submit just one after
tweaking its config, without touching the others.

## Output

- `job_out/<signal>_<jobid>.out` / `.err` -- LSF's own stdout/stderr capture
  (this is where the `train.py` print statements land).
- `checkpoints/<signal>/moment.pt` -- best checkpoint (lowest val loss),
  overwritten in place as training improves.
- `checkpoints/<signal>/history.csv` -- per-epoch train/val loss, flushed
  every epoch so it survives an interrupted job. Plot with:

  ```sh
  python3 plot_history.py checkpoints/<signal>/history.csv
  ```

Both `checkpoints/` and `job_out/` are gitignored, same as the original
vibration run -- copy whatever's worth keeping into `outputs/run_<jobid>/`
the way `outputs/run_29433679/` was archived, if you want it to survive a
clean checkout.

## Wall-time estimates

`train_vibration.sh`'s `-W 08:00` and `train_current.sh`'s are a rough 2x
safety margin over run_29433679's actual time (5980s for the old 3+5 = 8
total epochs), scaled up for the new epoch counts; `train_temp.sh` assumes
roughly vibration-like per-epoch cost. All three are guesses until a run
actually finishes -- adjust `-W` if a job gets killed for running over, or
tighten it once you know the real per-signal epoch time.
