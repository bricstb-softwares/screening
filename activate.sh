#!/bin/bash

mode=$1

if [ -d "$VIRTUALENV_NAMESPACE" ]; then
    echo "$VIRTUALENV_NAMESPACE exists."
    source $VIRTUALENV_NAMESPACE/bin/activate
else
    make
    source $VIRTUALENV_NAMESPACE/bin/activate
fi

export PATH=$PATH:$REPO_DIR/scripts
export PYTHONPATH=$PYTHONPATH:$REPO_DIR

if [ "$mode" == "jupyter" ]; then
    jupyter-lab --no-browser --port ${DOCKER_EXPOSE} --NotebookApp.token='' --NotebookApp.password=''
fi




