

maestro run slurm --device 0\
                  --max-procs 2\
                  --slurm-partition gpu-large\
                  --slurm-reservation joao.pinto_3\
                  --slurm-account joao.pinto\
                  --slurm-virtualenv /home/joao.pinto/git_repos/maestro/maestro-env\
                  --slurm-nodes 6\
