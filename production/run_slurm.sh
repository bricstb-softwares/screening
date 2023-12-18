#!/bin/bash

#SBATCH --job-name=base_shz_stc_mns
#SBATCH --output=logs/baseline/%x_job%j.log
#SBATCH --partition=gpu-large 		# put job into GPU partition
#SBATCH --exclusive			# request exclusive allocation of resources
#SBATCH --cpus-per-task=4		# number of CPUs per process
#SBATCH --account=philipp.gaspar
##SBATCH --reservation=philipp.gaspar_24

## nodes allocation
#SBATCH --nodes=1			# number of nodes
#SBATCH --ntasks-per-node=1		# MPI processes per node 

echo 'Nodelist:' $SLURM_JOB_NODELIST

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export IMG=/home/philipp.gaspar/BRICS-TB/images/screening_backdoor-gpu.sif
export BIND_PHIL=/home/philipp.gaspar/BRICS-TB
export BIND_BRICS=/home/brics/public

##srun -N 1 -n 1 -c $SLURM_CPUS_PER_TASK singularity run --nv --bind $BIND_PHIL --bind $BIND_BRICS $IMG -p altogether -info altogether_info.json -params hyperparameters.json 
srun -N 1 -n 1 -c $SLURM_CPUS_PER_TASK singularity exec --nv --bind $BIND_PHIL --bind $BIND_BRICS $IMG python ~/BRICS-TB/tb-brics-tools/screening/run.py -p baseline -info baseline_info.json -params hyperparameters.json
##srun -N 1 -n 1 -c $SLURM_CPUS_PER_TASK singularity exec  --nv --bind $BIND_PHIL --bind $BIND_BRICS $IMG -hash baseline -info report_info.json

wait

