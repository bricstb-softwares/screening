
#
# change this parameters
#
partition=gpu-large
reservation=joao.pinto_4
#target_path=/home/$USER/public
target_path=/home/joao.pinto/git_repos
max_procs=4
database_server_url=$POSTGRES_SERVER_URL # this is my postgres URL



#
# do no change after
#

local_path=$PWD

# setup maestro
if [ -d "$target_path/maestro" ]; then
    echo "$VIRTUALENV_NAMESPACE exists."
    cd $target_path/maestro
    source envs.sh
else # not exist
    cd $target_path
    git clone https://github.com/jodafons/maestro.git && cd maestro
    source envs.sh
fi

cd $local_path

maestro run slurm --device 0\
                  --max-procs $max_procs\
                  --slurm-partition $partition\
                  --slurm-reservation $reservation\
                  --slurm-account $USER\
                  --slurm-virtualenv $VIRTUAL_ENV\
                  --database-recreate \
                  --database-url $database_server_url \
                  --slurm-nodes 7\
                  --tracking-enable


cd $local_path