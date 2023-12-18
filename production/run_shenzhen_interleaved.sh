#!/bin/bash

#SBATCH --job-name=interleaved_shenzhen
#SBATCH --output=%x_job%j.log
#SBATCH --partition=gpu-large
#SBATCH --exclusive
#SBATCH --cpus-per-task=4
#SBATCH --account=philipp.gaspar

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

echo 'Nodelist:' $SLURM_JOB_NODELIST

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export IMG=/home/philipp.gaspar/BRICS-TB/screening_latest-gpu-NEW.sif
export BIND=/home/brics/public

srun -N 1 -n 1 -c $SLURM_CPUS_PER_TASK singularity run --nv --bind $BIND $IMG --process_name interleaved_shenzhen --config_file /app/config.json 

wait

