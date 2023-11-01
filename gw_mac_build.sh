#!/bin/bash

export BUILD_CUSTOM_PROTOBUF=1
export USE_PYTORCH_METAL=0
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py develop
