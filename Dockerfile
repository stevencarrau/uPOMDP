FROM ubuntu:22.04

# uPOMDP working dir
ENV MAIN_DIR "/usr/uPOMDP/uPOMDP"
# Set Storm dir for pystorm.
ENV STORM_DIR "/usr/uPOMDP/storm"
# Tensorflow env flag to get package to work with current requirements.
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION "python"

RUN set -eux && \
    apt-get update && \
    apt-get upgrade -yy && \
    apt-get install -yy build-essential git cmake libboost-all-dev libcln-dev libgmp-dev libginac-dev automake libglpk-dev libhwloc-dev libz3-dev libxerces-c-dev libeigen3-dev python3 python3-pip

RUN set -eux && \
    mkdir -p ${STORM_DIR} && \
    cd ${STORM_DIR} && \
    git clone --depth 1 -b 1.6.4 https://github.com/moves-rwth/storm.git . && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make -j4 && \
    make -j4 test

WORKDIR ${MAIN_DIR}

COPY . .

RUN set -eux && \
    python3 -m pip install -U pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install git+https://github.com/moves-rwth/pycarl.git@2.0.5 && \
    pip install git+https://github.com/moves-rwth/stormpy.git@1.6.4
