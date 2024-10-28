#!/bin/bash

source ${MODULESHOME}/init/bash
module reset
module load PrgEnv-amd cray-parallel-netcdf cmake craype-accel-amd-gfx90a cray-hdf5 cray-netcdf

export ROCM_PATH=${CRAY_AMD_COMPILER_PREFIX}

../../cmakeclean.sh

export CC=cc
export CXX=CC
export FC=ftn
export MPICH_GPU_SUPPORT_ENABLED=1

unset CXXFLAGS
unset CFLAGS
unset FFLAGS

cmake -Wno-dev                        \
      -DKokkos_ENABLE_HIP=ON          \
      -DKokkos_ARCH_AMD_GFX90A=ON     \
      -DYAKL_F90_FLAGS="-O3"          \
      -DCMAKE_INSTALL_PREFIX="`pwd`"  \
      -DYAKL_UNIT_CXX_FLAGS="-DYAKL_EXPERIMENTAL_HIP_LAUNCHER;-DHAVE_MPI;-DPORTURB_GPU_AWARE_MPI;-munsafe-fp-atomics;-O3;-ffast-math;-I${ROCM_PATH}/include;-D__HIP_ROCclr__;-D__HIP_ARCH_GFX90A__=1;-Wno-unused-result;-Wno-macro-redefined" \
      -DYAKL_UNIT_CXX_LINK_FLAGS="--rocm-path=${ROCM_PATH};-L${ROCM_PATH}/lib;-lamdhip64" \
      -DYAKL_TEST_NETCDF=ON           \
      -DMPI_COMMAND=""                \
      ../../..

