#!/bin/bash

source ${MODULESHOME}/init/bash
module reset
module load PrgEnv-gnu craype-network-ucx cray-mpich-ucx ucx/1.20.0 cray-hdf5 cray-netcdf
module load cuda/12.8 craype-accel-nvidia80

../../cmakeclean.sh

export CC=cc
export FC=ftn
export CXX=CC
export MPICH_GPU_SUPPORT_ENABLED=1

unset CXXFLAGS
unset FFLAGS
unset F77FLAGS
unset F90FLAGS

PNETCDF_DIR=/lustre/storm/nwp501/scratch/imn/pnetcdf_cray_mpich_ucx

export MPICH_GPU_SUPPORT_ENABLED=1
export UCX_TLS=rc,sm,self,cuda,cuda_copy,cuda_ipc

cmake -Wno-dev                          \
      -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
      -DKokkos_ENABLE_CUDA=ON           \
      -DKokkos_ARCH_AMPERE80=ON         \
      -DYAKL_F90_FLAGS="-O3"            \
      -DCMAKE_INSTALL_PREFIX="`pwd`"    \
      -DYAKL_UNIT_CXX_FLAGS="-DHAVE_MPI;-O1;-g" \
      -DYAKL_UNIT_CXX_LINK_FLAGS=""     \
      -DYAKL_TEST_NETCDF=ON             \
      -DMPI_COMMAND=""                  \
      ../../..

