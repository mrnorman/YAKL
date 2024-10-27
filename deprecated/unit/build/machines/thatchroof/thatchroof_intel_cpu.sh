#!/bin/bash

source /usr/share/modules/init/bash
module purge
module load icc

../../cmakeclean.sh

unset GATOR_DISABLE

export CC=icc
export CXX=icpc
export FC=ifort
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_CXX_FLAGS="-O3 -DYAKL_PROFILE -march=native -mtune=native -I`nc-config --includedir`" \
      -DYAKL_F90_FLAGS="-O3"                 \
      -DNETCDF_LINK_FLAGS="`nc-config --libs`"        \
      ../../..

