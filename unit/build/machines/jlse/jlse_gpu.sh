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
      -DCMAKE_CXX_FLAGS='-O0 -g -sycl-std=2020 -fsycl -fsycl-unnamed-lambda' \
      -DYAKL_F90_FLAGS="-O0 -g" \
      -DYAKL_C_FLAGS="-O0 -g"   \
      ../../..

