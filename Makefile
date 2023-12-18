
all: build_local
SHELL := /bin/bash



build_local:
	virtualenv -p python ${VIRTUALENV_NAMESPACE}
	source ${VIRTUALENV_NAMESPACE}/bin/activate && pip install poetry && poetry install --no-root

build_images:
	make build_base
	make build_prod 


build_base:
	docker build --network host --build-arg  --compress -t ${DOCKER_NAMESPACE}/screening:base -f base.Dockerfile .


build_prod:
	docker build --network host --build-arg DOCKER_NAMESPACE=${DOCKER_NAMESPACE} --build-arg GIT_SOURCE_COMMIT_ARG=$(git rev-parse HEAD) --compress -t ${DOCKER_NAMESPACE}/screening:prod -f prod.Dockerfile .


build_base_sif:
	docker push ${DOCKER_NAMESPACE}/screening:base
	singularity pull docker://${DOCKER_NAMESPACE}/screening:base
	mv *.sif ${PROJECT_DIR}


build_prod_sif:
	docker push ${DOCKER_NAMESPACE}/screening:prod
	singularity pull docker://${DOCKER_NAMESPACE}/screening:prod
	mv *.sif ${PROJECT_DIR}


run:
	singularity run --nv --bind=/home:/home  --bind=/mnt/brics_data:/mnt/brics_data --writable-tmpfs ${PROJECT_DIR}/images/screening_base.sif  

clean:
	docker system prune -a
	