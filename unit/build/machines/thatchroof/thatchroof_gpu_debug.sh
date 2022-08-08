#!/bin/bash

source /usr/share/modules/init/bash
module purge

../../cmakeclean.sh

unset GATOR_DISABLE

export CC=gcc
export CXX=g++
export FC=gfortran
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="CUDA"                                                                               \
      -DYAKL_CUDA_FLAGS="-O0 -g -DYAKL_DEBUG -DYAKL_ENABLE_STREAMS -DYAKL_VERBOSE_FILE -arch sm_86 -ccbin g++ -I`nc-config --includedir`" \
      -DYAKL_F90_FLAGS="-O0 -g"                                                                           \
      -DNETCDF_LINK_FLAGS="`nc-config --libs`"        \
      ../../..

