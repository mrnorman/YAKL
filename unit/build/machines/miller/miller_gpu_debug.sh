#!/bin/bash

source ${MODULESHOME}/init/bash
module reset
module load PrgEnv-gnu cuda/12.6 craype-accel-nvidia80 cray-hdf5 cray-netcdf cray-parallel-netcdf

../../cmakeclean.sh

export CC=cc
export FC=ftn
export CXX=CC

unset CXXFLAGS
unset FFLAGS
unset F77FLAGS
unset F90FLAGS

cmake -Wno-dev                          \
      -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
      -DKokkos_ENABLE_CUDA=ON           \
      -DKokkos_ARCH_AMPERE80=ON        \
      -DYAKL_F90_FLAGS="-O0"            \
      -DCMAKE_INSTALL_PREFIX="`pwd`"    \
      -DYAKL_UNIT_CXX_FLAGS="-O0;-g" \
      -DYAKL_UNIT_CXX_LINK_FLAGS=""     \
      -DYAKL_TEST_NETCDF=ON            \
      -DYAKL_TEST_PNETCDF=ON           \
      -DMPI_COMMAND=""                  \
      ../../..

