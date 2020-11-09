#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load DefApps gcc/8.1.1 cmake/3.15.2 cuda/10.1.105

./cmakeclean.sh

unset GATOR_DISABLE

export CC=mpicc
export CXX=mpic++
export CXXFLAGS="-O3"
export FFLAGS="-O3"

cmake -DARCH="CUDA"                  \
      -DCUDA_FLAGS="-O3 -arch sm_70 -ccbin mpic++" \
      ..

