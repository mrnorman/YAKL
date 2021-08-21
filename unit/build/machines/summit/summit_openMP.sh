#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load DefApps xl cuda cmake

./cmakeclean.sh

unset GATOR_DISABLE

export CC=mpicc
export CXX=mpic++
unset CXXFLAGS
export FFLAGS="-O3"

cmake -DYAKL_ARCH="OPENMP45" \
      -DYAKL_OPENMP45_FLAGS="-O3 -qsmp=omp -qoffload" \
      ..

