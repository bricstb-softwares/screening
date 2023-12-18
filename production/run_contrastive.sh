#!/bin/bash

#SBATCH --job-name=contrastive
#SBATCH --output=%x_job%j.log
#SBATCH --partition=gpu-large
#SBATCH --exclusive
#SBATCH --account=philipp.gaspar
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

export HOME=/home/philipp.gaspar/BRICS-TB
export IMG=$HOME/images/contrastive_gpu.sif
export PACKAGE=$HOME/tb-brics-tools

srun singularity exec --nv $IMG python $PACKAGE/contrastive/run.py -p TrainBRICSNet -c $PACKAGE/contrastive/config.yaml
