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

cmake -DYAKL_CXX_FLAGS="-O0 -g -DYAKL_DEBUG"     \
      -DYAKL_F90_FLAGS="-O0 -g"                  \
      -DYAKL_C_FLAGS="-O0 -g"                    \
      ../../..

