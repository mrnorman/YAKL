#!/bin/bash

./cmakeclean.sh

unset GATOR_DISABLE

export CC=gcc
export CXX=g++
export CXXFLAGS="-O3"
export FFLAGS="-O3"

export CXXFLAGS="-O3"

cmake -DARCH="CUDA"                             \
      -DCUDA_FLAGS="-O3 -arch sm_50 -ccbin g++" \
      ..

