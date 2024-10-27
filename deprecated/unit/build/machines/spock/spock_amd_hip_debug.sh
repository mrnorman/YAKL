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
      -DYAKL_HIP_FLAGS="-O0 -g -DYAKL_DEBUG --offload-arch=gfx908 -x hip" \
      -DYAKL_F90_FLAGS="-O0 -g"                            \
      -DYAKL_C_FLAGS="-O0 -g"                              \
      -DMPI_COMMAND="" \
      ../../..

