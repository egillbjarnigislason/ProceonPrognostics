#!/bin/sh
### Current fine-tune (new instance): Butterworth low-pass at 800 Hz --
### see config.py:current_config for why 800 Hz rather than the 12000 Hz
### that was sitting in the notebook.
#BSUB -q gpua100
#BSUB -J moment_finetune_current
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 08:00
#BSUB -o job_out/current_%J.out
#BSUB -e job_out/current_%J.err

mkdir -p job_out
module load python3/3.11.13
source .venv/bin/activate

python3 train.py --signal current
