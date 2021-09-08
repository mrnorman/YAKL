#!/bin/bash

../../cmakeclean.sh

unset GATOR_DISABLE

export CC=gcc
export CXX=g++
export FC=gfortran
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="CUDA"                                                                               \
      -DYAKL_CUDA_FLAGS="-O0 -g -DYAKL_DEBUG -arch sm_35 -ccbin g++ -DTHRUST_IGNORE_CUB_VERSION_CHECK -I/opt/netcdf_gnu/include" \
      -DYAKL_C_FLAGS="-O3"                                                                             \
      -DYAKL_F90_FLAGS="-O3"                                                                           \
      -DNETCDF_LINK_FLAGS="`/opt/netcdf_gnu/bin/nc-config --libs`"        \
      ../../..

