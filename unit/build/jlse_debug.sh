#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load oneapi cmake

./cmakeclean.sh

unset GATOR_DISABLE

export CC=icx
export CXX=icpx
export FC=ifx
unset CXXFLAGS
export FFLAGS="-O0 -g"

cmake -DYAKL_CXX_FLAGS="-O0 -g -DYAKL_DEBUG" ..

