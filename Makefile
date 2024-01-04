
all: build_local
SHELL := /bin/bash




build_local:
	virtualenv -p python ${VIRTUALENV_NAMESPACE}
	source ${VIRTUALENV_NAMESPACE}/bin/activate && pip install --upgrade pip && pip install -e .


build_base:
	docker build --network host --build-arg  --compress -t ${DOCKER_NAMESPACE}/screening:base -f Dockerfile .


build_sif:
	docker push ${DOCKER_NAMESPACE}/screening:base
	singularity pull docker://${DOCKER_NAMESPACE}/screening:base
	mv *.sif ${PROJECT_DIR}/images


run:
	singularity run --nv --bind=/home:/home  --bind=${PROJECT_DIR}:${PROJECT_DIR} --writable-tmpfs ${PROJECT_DIR}/images/screening_base.sif  


clean:
	docker system prune -a
	