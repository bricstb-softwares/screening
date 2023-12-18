ARG GIT_SOURCE_COMMIT_ARG
ARG DOCKER_NAMESPACE
# NOTE: do not change this image base
FROM ${DOCKER_NAMESPACE}/screening:base
LABEL maintainer="philipp.gaspar@gmail.com"


ENV GIT_SOURCE_COMMIT=${GIT_SOURCE_COMMIT_ARG}


WORKDIR /app
COPY pipelines .
COPY tasks .
COPY utils .
COPY run.py .
COPY run_report.py .
COPY requirements.txt .

RUN pip install --upgrade pip 
RUN pip install -r requirements.txt


#ENTRYPOINT [ "python", "/app/run.py" ]