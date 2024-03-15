#!/bin/bash

# These commands should be run at init and esp. after rebase, within a custom conda environment
git submodule sync
git submodule update --init --recursive
conda install cmake ninja
python -m pip install -r requirements.txt
conda install pkg-config libuv

# Here is the main build
export BUILD_CUSTOM_PROTOBUF=1
export USE_PYTORCH_METAL=0
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
DEBUG=1 python setup.py develop
