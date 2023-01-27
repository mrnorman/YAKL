#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load oneapi/eng-compiler/2022.12.30.002
module use /lus/gila/projects/Aurora_deployment/intel_anl_shared_CNDA/intel-gpu-umd/modulefiles/intel_compute_runtime/release
module unload intel_compute_runtime/release/agama-devel-524
module load 2023.01.25-5b66ab0
module load spack cmake
module list

../../cmakeclean.sh

unset GATOR_DISABLE

export CC=mpicc
export CXX=mpicxx
export FC=mpifort
unset CXXFLAGS
unset FFLAGS

export ONEAPI_MPICH_GPU=NO_GPU
export ZE_AFFINITY_MASK=0.0

cmake -DYAKL_ARCH="SYCL"        \
      -DYAKL_SYCL_FLAGS="-O1 -g -Wsycl-strict -DYAKL_DEBUG" \
      -DCMAKE_CXX_FLAGS="-O1 -g -fsycl -sycl-std=2020 -fsycl-unnamed-lambda -fsycl-device-code-split=per_kernel" \
      -DYAKL_F90_FLAGS="-O0 -g" \
      -DYAKL_C_FLAGS="-O0 -g"   \
      ../../..
