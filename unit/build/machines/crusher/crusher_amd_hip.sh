#!/bin/bash

source $MODULESHOME/init/bash
module load PrgEnv-amd craype-accel-amd-gfx90a rocm

../../cmakeclean.sh

unset GATOR_DISABLE

export CC=cc
export CXX=CC
export FC=ftn
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="HIP"                                \
      -DYAKL_HIP_FLAGS="-O3 -DYAKL_ENABLE_STREAMS -Wno-tautological-pointer-compare -Wno-unused-result -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=${ROCM_PATH} --offload-arch=gfx90a -x hip" \
      -DYAKL_F90_FLAGS="-O3"                            \
      -DCMAKE_EXE_LINKER_FLAGS="--rocm-path=${ROCM_PATH} -L${ROCM_PATH}/lib -lamdhip64" \
      -DMPI_COMMAND="" \
      ../../..

