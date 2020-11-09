#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load rocm cmake

./cmakeclean.sh

unset GATOR_DISABLE

export CC=hipcc
export CXX=hipcc
export CXXFLAGS="-O3"
export FFLAGS="-O3"

cmake -DARCH="HIP"                  \
      -DHIP_FLAGS="-O3" \
      -Wno-dev  \
      ..

