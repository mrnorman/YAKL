#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load DefApps gcc/9.3.0 cmake

../../cmakeclean.sh

unset GATOR_DISABLE

export CC=mpicc
export CXX=mpic++
export FC=mpif90
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_CXX_FLAGS="-O0 -g -DYAKL_DEBUG"    \
      -DYAKL_F90_FLAGS="-O0 -g"                 \
      -DYAKL_C_FLAGS="-O0 -g"                   \
      -DMPI_COMMAND="jsrun -n 1 -c 1 -a 1 -g 1" \
      ../../..

