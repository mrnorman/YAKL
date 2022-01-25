#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load oneapi cmake

../../cmakeclean.sh

unset GATOR_DISABLE

export CC=mpiicx
export CXX=mpiicpx
export FC=mpiifx
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="SYCL"        \
      -DYAKL_SYCL_FLAGS="-O0 -g -DYAKL_DEBUG" \
      -DCMAKE_CXX_FLAGS="-O0 -g -fsycl -sycl-std=2020 -fsycl-unnamed-lambda -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations" \
      -DYAKL_F90_FLAGS="-O0 -g" \
      -DYAKL_C_FLAGS="-O0 -g"   \
      ../../..

