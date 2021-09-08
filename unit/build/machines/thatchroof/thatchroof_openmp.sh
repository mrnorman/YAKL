#!/bin/bash

../../cmakeclean.sh

unset GATOR_DISABLE
export OMP_NUM_THREADS=24

export CC=gcc
export CXX=g++
export FC=gfortran
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="OPENMP"               \
      -DYAKL_OPENMP_FLAGS="-O3 -fopenmp -I/opt/netcdf_gnu/include" \
      -DYAKL_C_FLAGS="-O3"               \
      -DYAKL_F90_FLAGS="-O3"             \
      -DNETCDF_LINK_FLAGS="`/opt/netcdf_gnu/bin/nc-config --libs`"        \
      ../../..

