#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load rocm cmake

./cmakeclean.sh

unset GATOR_DISABLE

export CC=hipcc
export CXX=hipcc
unset CXXFLAGS
export FFLAGS="-O3"

cmake -DYAKL_ARCH="HIP"      \
      -DYAKL_HIP_FLAGS="-O3" \
      -Wno-dev               \
      ..

