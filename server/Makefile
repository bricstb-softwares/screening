
SHELL := /bin/bash


all: build

#
# Build
#
build: 
	docker build --progress=plain -t ${DOCKER_NAMESPACE}/cad-server --compress .
	
up:
	docker compose up -d

up_debug:
	docker compose up

down:
	docker compose down

clean:
	docker system prune -a

