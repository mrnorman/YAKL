#!/bin/bash

source $MODULESHOME/init/bash
module load PrgEnv-amd craype-accel-amd-gfx908

../../cmakeclean.sh

unset GATOR_DISABLE

export CC=cc
export CXX=CC
export FC=ftn
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="HIP"                                \
      -DYAKL_HIP_FLAGS="-O3" \
      -DYAKL_F90_FLAGS="-O3"                            \
      -DYAKL_C_FLAGS="-O3"                              \
      -DMPI_COMMAND="" \
      ../../..

