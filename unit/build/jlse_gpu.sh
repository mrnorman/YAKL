#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load oneapi cmake

./cmakeclean.sh

unset GATOR_DISABLE

export CC=icx
export CXX=icpx
export FC=ifx
export CXXFLAGS="-O0 -g"
export FFLAGS="-O0 -g"

cmake -DARCH="SYCL"                     \
      -DSYCL_FLAGS="-O0 -g --intel -fsycl" \
      ..
