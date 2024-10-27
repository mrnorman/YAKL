#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load rocm cmake

../../cmakeclean.sh

unset GATOR_DISABLE

export CC=hipcc
export CXX=hipcc
export FC=gfortran
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="HIP"          \
      -DYAKL_HIP_FLAGS="-O3"     \
      -DYAKL_F90_FLAGS="-O3"     \
      -DYAKL_C_FLAGS="-O3"       \
      ../../..

#      -Wno-dev               \
