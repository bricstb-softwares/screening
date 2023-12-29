FROM tensorflow/tensorflow:2.10.0-gpu
#FROM tensorflow/tensorflow:2.15.0-gpu

LABEL maintainer="philipp.gaspar@gmail.com"

RUN apt-get update && apt-get install -y git
RUN pip install --upgrade pip
RUN pip install virtualenv poetry


