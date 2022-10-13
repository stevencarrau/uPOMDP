# uPOMDP
Uncertain POMDP planning setup


## getting started

Instructions tested on Ubuntu 22.04.

1. setup working directory
   ```bash
    export WORKSPACE=~/workspace
    mkdir $WORKSPACE
    ```
2. install dependencies for the storm model checker (instructions for other systems can be found [here](https://www.stormchecker.org/documentation/obtain-storm/dependencies.html))
   ```bash
   sudo apt-get install build-essential git cmake libboost-all-dev libcln-dev libgmp-dev libginac-dev automake libglpk-dev libhwloc-dev
   ```
3. install storm 1.6.4
    ```bash
    cd $WORKSPACE
    git clone --depth 1 -b 1.6.4 https://github.com/moves-rwth/storm.git
    export STORM_DIR=storm
    cd $STORM_DIR
    mkdir build
    cd build
    cmake ..
    make
    ```
4. set up virtual environment and install python packages
    ```bash
    cd $WORKSPACE
    git clone https://github.com/stevencarrau/uPOMDP
    cd uPOMDP/
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install -r requirements.txt
    ```
5. install python bindings for storm
    ```bash
    pip install git+https://github.com/moves-rwth/pycarl.git@2.0.5
    pip install git+https://github.com/moves-rwth/stormpy.git@1.6.4
    ```
