
all: build_local
SHELL := /bin/bash


build_local:
	virtualenv -p python ${VIRTUALENV_NAMESPACE}
	source ${VIRTUALENV_NAMESPACE}/bin/activate && pip install --upgrade pip && pip install -r requirements.txt


build_base:
	docker build --network host --build-arg  --compress -t ${DOCKER_NAMESPACE}/screening:base -f Dockerfile .


build_sif:
	docker push ${DOCKER_NAMESPACE}/screening:base
	singularity pull docker://${DOCKER_NAMESPACE}/screening:base
	mv *.sif ${PROJECT_DIR}/images


run:
	singularity run --nv --bind=/home:/home --bind=${PROJECT_DIR}:${PROJECT_DIR} --writable-tmpfs ${PROJECT_DIR}/images/screening_base.sif  

run_jupyter:
	singularity exec --nv --bind=/home:/home --bind=${PROJECT_DIR}:${PROJECT_DIR} --writable-tmpfs ${PROJECT_DIR}/images/screening_base.sif /bin/bash ./activate.sh jupyter

clean:
	docker system prune -a

	

run_maestro:

	source ${MAESTRO_VIRTUALENV_PATH}/bin/activate
	maestro slurm cancel
	rm *.out && rm *.log
	maestro run slurm --device auto\
                      --message-level INFO\
                      --max-procs 4\
                      --slurm-partition gpu-large\
                      --slurm-reservation ${SLURM_RESERVATION}\
                      --slurm-account ${USER}\
                      --slurm-virtualenv ${MAESTRO_VIRTUALENV_PATH}\
                      --database-recreate \
                      --database-url ${POSTGRES_SERVER_URL} \
                      --slurm-nodes 7\
