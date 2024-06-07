FROM python:3.10-alpine

LABEL description="Container with all necessary tools for running the language intermodulation \
    experiment, ingesting acquired data from MEG, MRI, and behavioral experiments, and running \
    analyses on the data."
LABEL maintainer="Berk Gercek (@github:berkgercek)"
LABEL version="0.1"

WORKDIR /data
COPY ./ /data/
RUN apk update && apk add bash git wget build-base
# gfortran pkgconfig cmake
SHELL ["/bin/bash", "-c"]
RUN pip install -U pip
RUN pip install -e .[analysis]
