#!/bin/bash

set -e
set -x

mkdir -p gw_out

export BUILD_CUSTOM_PROTOBUF=1
export USE_PYTORCH_METAL=0
export CXX=/usr/bin/g++
export CC=/usr/bin/cc
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export USE_MPS=0
export USE_APU=1

# locate brew's cmake because conda's is now broken
export PATH=/usr/local/bin:$PATH 

DEBUG=1 python setup.py develop | tee gw_out/setup_log.txt
