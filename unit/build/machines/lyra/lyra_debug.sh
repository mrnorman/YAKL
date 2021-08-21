#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load rocm cmake

./cmakeclean.sh

unset GATOR_DISABLE

export CC=hipcc
export CXX=hipcc
unset CXXFLAGS
export FFLAGS="-O0 -g"

cmake -DYAKL_CXX_FLAGS="-O0 -g -DYAKL_DEBUG" ..

