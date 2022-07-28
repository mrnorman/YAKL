#!/bin/bash

source /usr/share/modules/init/bash
module purge

../../cmakeclean.sh

unset GATOR_DISABLE

export CC=gcc-11
export CXX=g++-11
export FC=gfortran-11
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_CXX_FLAGS="-Ofast -DYAKL_PROFILE -march=native -mtune=native -I`nc-config --includedir`" \
      -DYAKL_F90_FLAGS="-O3"                 \
      -DNETCDF_LINK_FLAGS="`nc-config --libs`"        \
      ../../..

