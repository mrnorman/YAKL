#!/bin/bash

source ${MODULESHOME}/init/bash
module reset
module load PrgEnv-nvidia cray-hdf5 cray-netcdf cray-libsci/25.03.0 cray-parallel-netcdf
module load cudatoolkit craype-accel-nvidia90

../../cmakeclean.sh

export CC=cc
export FC=ftn
export CXX=CC

unset CXXFLAGS
unset FFLAGS
unset F77FLAGS
unset F90FLAGS

export MPICH_GPU_SUPPORT_ENABLED=1

cmake -Wno-dev                          \
      -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
      -DKokkos_ENABLE_CUDA=ON           \
      -DKokkos_ARCH_HOPPER90=ON        \
      -DYAKL_F90_FLAGS="-O3"            \
      -DCMAKE_INSTALL_PREFIX="`pwd`"    \
      -DYAKL_UNIT_CXX_FLAGS="-DHAVE_MPI;-O0;-g;-G;-Wextra;-traceback;-DYAKL_AUTO_FENCE;-fno-omit-frame-pointer" \
      -DYAKL_UNIT_CXX_LINK_FLAGS=""     \
      -DYAKL_TEST_NETCDF=ON             \
      -DYAKL_TEST_PNETCDF=ON            \
      -DMPI_COMMAND=""                  \
      ../../..

