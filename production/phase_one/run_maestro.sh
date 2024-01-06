

maestro run slurm --device 0\
                  --max-procs 2\
                  --slurm-partition gpu-large\
                  --slurm-reservation joao.pinto_3\
                  --slurm-account joao.pinto\
                  --slurm-virtualenv $VIRTUAL_ENV\
                  --database-recreate \
                  --slurm-nodes 7\
                  