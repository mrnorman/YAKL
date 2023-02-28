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

cmake -DYAKL_CXX_FLAGS="-O0 -g -DYAKL_DEBUG -DYAKL_VERBOSE_FILE -I`nc-config --includedir`" \
      -DYAKL_F90_FLAGS="-O3"                 \
      -DNETCDF_LINK_FLAGS="`nc-config --libs`"        \
      -DCMAKE_INSTALL_PREFIX="`pwd`" \
      -DYAKL_TARGET_SUFFIX="debug" \
      ../../..

