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

export OMP_NUM_THREADS=64

cmake -Wno-dev                               \
      -DKokkos_ENABLE_OPENMP=ON              \
      -DKokkos_ARCH_NATIVE=ON                \
      -DYAKL_F90_FLAGS="-O3"                 \
      -DCMAKE_INSTALL_PREFIX="`pwd`"         \
      -DYAKL_UNIT_CXX_FLAGS="-DHAVE_MPI;-fopenmp;-O3;-Wno-unused-result;-Wno-macro-redefined" \
      -DYAKL_UNIT_CXX_LINK_FLAGS="-fopenmp"  \
      -DYAKL_TEST_NETCDF=ON                  \
      -DMPI_COMMAND=""                       \
      ../../..

