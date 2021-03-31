#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load DefApps xl cuda cmake

./cmakeclean.sh

unset GATOR_DISABLE

export CC=mpicc
export CXX=mpic++
export CXXFLAGS="-O3 -qsmp=omp -qoffload"
export FFLAGS="-O3"


cmake -DARCH="OPENMP45" \
      -DOPENMP45_FLAGS="" \
      ..

