#!/bin/bash

../../cmakeclean.sh

export CC=gcc-15
export CXX=g++-15
export FC=gfortran-15

unset CXXFLAGS
unset CFLAGS
unset FFLAGS

cmake -Wno-dev                        \
      -DYAKL_F90_FLAGS="-O3"          \
      -DCMAKE_INSTALL_PREFIX="`pwd`"  \
      -DYAKL_UNIT_CXX_FLAGS="-O3;-Wno-unused-result;-Wno-macro-redefined" \
      -DYAKL_UNIT_CXX_LINK_FLAGS=""   \
      -DYAKL_TEST_NETCDF=OFF          \
      -DMPI_COMMAND=""                \
      ../../..

