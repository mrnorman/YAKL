#!/bin/bash

../../cmakeclean.sh

unset GATOR_DISABLE

export CC=gcc
export CXX=g++
export FC=gfortran
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="CUDA"                                                                               \
      -DYAKL_CUDA_FLAGS="-O0 -g -DYAKL_DEBUG -arch sm_61 -ccbin g++" \
      -DYAKL_C_FLAGS="-O3"                                                                             \
      -DYAKL_F90_FLAGS="-O3"                                                                           \
      -DNETCDF_LINK_FLAGS="`nc-config --libs`"        \
      ../../..

