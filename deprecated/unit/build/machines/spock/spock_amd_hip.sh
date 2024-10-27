#!/bin/bash

source $MODULESHOME/init/bash
module load PrgEnv-amd craype-accel-amd-gfx908 rocm gcc

../../cmakeclean.sh

unset GATOR_DISABLE

export CC=hipcc
export CXX=hipcc
export FC=gfortran
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="HIP"                                \
      -DYAKL_HIP_FLAGS="-O3 --offload-arch=gfx908 -x hip" \
      -DYAKL_F90_FLAGS="-O3"                            \
      -DYAKL_C_FLAGS="-O3"                              \
      -DMPI_COMMAND="" \
      ../../..

