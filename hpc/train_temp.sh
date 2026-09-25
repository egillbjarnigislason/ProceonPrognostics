#!/bin/sh
### Temperature fine-tune (new instance): moving-average smoothing,
### window=4000 samples -- see config.py:temp_config.
#BSUB -q gpua100
#BSUB -J moment_finetune_temp
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 06:00
#BSUB -o job_out/temp_%J.out
#BSUB -e job_out/temp_%J.err

mkdir -p job_out
module load python3/3.11.13
source .venv/bin/activate

python3 train.py --signal temp
