#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load DefApps gcc/9.3.0 cmake

../../cmakeclean.sh

unset GATOR_DISABLE
export OMP_NUM_THREADS=24

export CC=mpicc
export CXX=mpic++
export FC=mpif90
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="OPENMP"               \
      -DYAKL_OPENMP_FLAGS="-O3 -fopenmp" \
      -DYAKL_F90_FLAGS="-O3"             \
      -DYAKL_C_FLAGS="-O3"               \
      -DMPI_COMMAND="jsrun -n 1 -c 1 -a 1 -g 1" \
      ../../..

