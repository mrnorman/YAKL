#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load oneapi cmake/3.22.1

../../cmakeclean.sh

unset GATOR_DISABLE

export CC=mpicc
export CXX=mpicxx
export FC=mpifort
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="SYCL" \
      -DYAKL_SYCL_FLAGS="-O3" \
      -DCMAKE_CXX_FLAGS="-O3 -fsycl -sycl-std=2020 -fsycl-unnamed-lambda -fsycl-device-code-split=per_kernel" \
      -DYAKL_F90_FLAGS="-O3" \
      -DYAKL_C_FLAGS="-O3"   \
      ../../..

