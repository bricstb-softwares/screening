FROM tensorflow/tensorflow:2.11.0-gpu
#FROM tensorflow/tensorflow:2.15.0-gpu
LABEL maintainer="philipp.gaspar@gmail.com"


ENV LC_ALL C.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV TERM screen
ENV TZ=America/New_York
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git
#RUN apt-get install msttcorefonts -qq
RUN apt-get install -y texlive-full
#RUN pip install --upgrade pipnvcc

RUN pip install virtualenv poetry



