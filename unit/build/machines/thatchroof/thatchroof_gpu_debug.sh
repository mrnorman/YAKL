#!/bin/bash

../../cmakeclean.sh

export CC=gcc
export CXX=mpic++
export FC=gfortran

unset CXXFLAGS
unset CFLAGS
unset FFLAGS
unset FCFLAGS

cmake -Wno-dev                              \
      -DKokkos_ENABLE_CUDA=ON               \
      -DKokkos_ARCH_AMPERE86=ON             \
      -DKokkos_ENABLE_DEBUG=ON              \
      -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON \
      -DKokkos_ENABLE_CUDA_CONSTEXPR=ON     \
      -DYAKL_F90_FLAGS="-O0;-g"             \
      -DCMAKE_INSTALL_PREFIX="`pwd`"        \
      -DYAKL_UNIT_CXX_FLAGS="-O0;-g;-Wno-unused-result;-Wno-macro-redefined" \
      -DYAKL_UNIT_CXX_LINK_FLAGS="-g;-lnetcdf;-lpnetcdf" \
      -DYAKL_TEST_NETCDF=ON                 \
      -DYAKL_TEST_PNETCDF=ON                \
      -DMPI_COMMAND=""                      \
      ../../..

