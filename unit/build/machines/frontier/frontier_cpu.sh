#!/bin/bash

source ${MODULESHOME}/init/bash
module reset
module load PrgEnv-gnu cray-parallel-netcdf cmake cray-hdf5 cray-netcdf

export CRAYPE_LINK_TYPE=dynamic

../../cmakeclean.sh

export CC=cc
export CXX=CC
export FC=ftn

unset CXXFLAGS
unset CFLAGS
unset FFLAGS

cmake -Wno-dev                        \
      -DYAKL_F90_FLAGS="-O3"          \
      -DCMAKE_INSTALL_PREFIX="`pwd`"  \
      -DYAKL_UNIT_CXX_FLAGS="-DHAVE_MPI;-O3;-Wno-unused-result;-Wno-macro-redefined" \
      -DYAKL_UNIT_CXX_LINK_FLAGS=""   \
      -DYAKL_TEST_NETCDF=ON           \
      -DMPI_COMMAND=""                \
      ../../..

