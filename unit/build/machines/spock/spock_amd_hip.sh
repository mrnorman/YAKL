#!/bin/bash

source $MODULESHOME/init/bash
module load PrgEnv-cray craype-accel-amd-gfx908 rocm

../../cmakeclean.sh

unset GATOR_DISABLE

export CC=cc
export CXX=CC
export FC=$GCC_X86_64/bin/gfortran
unset CXXFLAGS
unset FFLAGS
export LD_LIBRARY_PATH=$GCC_X86_64/lib64:$LD_LIBRARY_PATH

cmake -DYAKL_ARCH="HIP"                                \
      -DYAKL_HIP_FLAGS="-O3 -D__HIP_ROCclr__ -D__HIP_ARCH_GFX908__=1 --rocm-path=${ROCM_PATH} --offload-arch=gfx908 -x hip" \
      -DYAKL_F90_FLAGS="-O3"                            \
      -DYAKL_C_FLAGS="-O3"                              \
      -DCMAKE_EXE_LINKER_FLAGS="-L${HIP_PATH}/lib -lamdhip64" \
      -DMPI_COMMAND="" \
      ../../..

