#!/bin/bash

#SBATCH --job-name=baseline
#SBATCH --output=logs/screening/%x_job%j.log
#SBATCH --partition=gpu-large
#SBATCH --exclusive
#SBATCH --account=philipp.gaspar
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

export HOME=/home/philipp.gaspar/BRICS-TB
export BRICS=/home/brics/public
export IMG=$HOME/images
export PACKAGE=$HOME/tb-brics-tools

srun singularity exec --nv --bind $BRICS $IMG/screening_backdoor-gpu.sif python $PACKAGE/screening/run.py -p baseline -info baseline_info.json -params hyperparameters.json


