FROM ubuntu:22.04

WORKDIR /usr/uPOMDP

ENV STORM_DIR "/usr/uPOMDP/storm"

RUN set -eux && \
    apt-get update && \
    apt-get upgrade -yy && \
    apt-get install -yy build-essential git cmake libboost-all-dev libcln-dev libgmp-dev libginac-dev automake libglpk-dev libhwloc-dev libz3-dev libxerces-c-dev libeigen3-dev python3 python3-pip

RUN set -eux && \
    git clone --depth 1 -b 1.6.4 https://github.com/moves-rwth/storm.git && \
    cd ${STORM_DIR} && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make -j4 && \
    make test

RUN set -eux && \
    git clone https://github.com/stevencarrau/uPOMDP && \
    cd uPOMDP/ && \
    python3 -m pip install -U pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install git+https://github.com/moves-rwth/pycarl.git@2.0.5 && \
    pip install git+https://github.com/moves-rwth/stormpy.git@1.6.4
