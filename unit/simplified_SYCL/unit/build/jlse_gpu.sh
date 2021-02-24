#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load oneapi cmake

./cmakeclean.sh

unset GATOR_DISABLE

export CC=icc
export CXX=icpx
export CXXFLAGS="-O0 -g"
export FFLAGS="-O0 -g"

cmake -DARCH="SYCL"                     \
      -DSYCL_FLAGS="-O0 -g --intel -fsycl" \
      ..
