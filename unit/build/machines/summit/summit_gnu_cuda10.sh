#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load DefApps gcc/7.5.0 cuda/10.1.168 cmake

../../cmakeclean.sh

unset GATOR_DISABLE

export CC=mpicc
export CXX=mpic++
export FC=mpif90
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="CUDA"                                \
      -DYAKL_CUDA_FLAGS="-O3 -arch sm_70 -ccbin mpic++ -DTHRUST_IGNORE_CUB_VERSION_CHECK" \
      -DYAKL_F90_FLAGS="-O3"                            \
      -DYAKL_C_FLAGS="-O3"                              \
      -DMPI_COMMAND="jsrun -n 1 -c 1 -a 1 -g 1" \
      ../../..

