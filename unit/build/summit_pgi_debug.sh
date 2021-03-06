#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load DefApps pgi cmake

./cmakeclean.sh

unset GATOR_DISABLE

export CC=mpicc
export CXX=mpic++
unset CXXFLAGS
export FFLAGS="-O0 -g"

cmake -DYAKL_CXX_FLAGS="-O0 -g -DYAKL_DEBUG" ..

