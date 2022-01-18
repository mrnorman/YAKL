#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load oneapi cmake

../../cmakeclean.sh

unset GATOR_DISABLE

export CC=icx
export CXX=icpx
export FC=ifx
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="SYCL"        \
      -DYAKL_SYCL_FLAGS="-O1 -fsycl" \
      -DCMAKE_CXX_FLAGS="-O1 -sycl-std=2020 -fsycl-unnamed-lambda" \
      -DYAKL_F90_FLAGS="-O3" \
      -DYAKL_C_FLAGS="-O3"   \
      ../../..

