#!/bin/bash

../../cmakeclean.sh

unset GATOR_DISABLE

export CC=gcc
export CXX=g++
export FC=gfortran
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_CXX_FLAGS="-O0 -g -DYAKL_DEBUG"  \
      -DYAKL_F90_FLAGS="-O0 -g"  \
      -DYAKL_C_FLAGS="-O0 -g"    \
      ../../..

