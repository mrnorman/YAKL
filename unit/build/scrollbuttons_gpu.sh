#!/bin/bash

./cmakeclean.sh

export CXXFLAGS="-O3"

cmake -DARCH="CUDA"                  \
      -DCUDA_FLAGS="-O3 -arch sm_61" \
      ..

