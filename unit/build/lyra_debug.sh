#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load rocm cmake

./cmakeclean.sh

unset GATOR_DISABLE

export CC=hipcc
export CXX=hipcc
export CXXFLAGS="-O0 -g -DYAKL_DEBUG"
export FFLAGS="-O0 -g"

cmake ..

