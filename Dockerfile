FROM ubuntu:24.04

LABEL description="Container with all necessary tools for analyzing the language intermodulation \
    experiment: ingesting acquired data from MEG, MRI, and behavioral info, then running \
    analyses on the data."
LABEL maintainer="Berk Gercek (@github:berkgercek)"
LABEL version="0.1"

WORKDIR /data
COPY ./ /data/
ENV DEBIAN_FRONTEND=noninteractive \
    DEBCONF_NONINTERACTIVE_SEEN=true
RUN apt update && apt install -y build-essential wget
RUN wget -O Miniforge3.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
RUN bash Miniforge3.sh -b -p "/opt/conda"
# gfortran pkgconfig cmake
SHELL ["/bin/bash", "-c"]
RUN /bin/bash -c "/opt/conda/bin/conda create -n intermod python=3.10 &&\
    source /opt/conda/bin/activate intermod &&\
    conda install wxpython &&\
    pip install -e /data[analysis] &&\
    conda clean --all -f -y"
ENTRYPOINT [ "/data/docker/entrypoint.sh" ]

