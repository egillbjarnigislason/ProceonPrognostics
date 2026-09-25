#!/bin/sh
### Vibration fine-tune (repeat of run_29433679, with the Butterworth
### denoise from NB_DataExpl.ipynb folded in and phase 2 extended to 16
### epochs -- see config.py:vibration_config).
#BSUB -q gpua100
#BSUB -J moment_finetune_vibration
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 08:00
#BSUB -o job_out/vibration_%J.out
#BSUB -e job_out/vibration_%J.err

mkdir -p job_out
module load python3/3.11.13
source .venv/bin/activate

python3 train.py --signal vibration
