#!/bin/bash

source /usr/share/modules/init/bash
module purge
module load cmake-3.23.2-gcc-11.1.0-kvgnqc6 netcdf-c-4.9.2-gcc-11.1.0-mvu6i6y

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
      -DCMAKE_INSTALL_PREFIX="`pwd`" \
      -DYAKL_TARGET_SUFFIX="debug" \
      -DNETCDF_LINK_FLAGS="`nc-config --libs`"        \
      ../../..

