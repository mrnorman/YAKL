#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load spack cmake ninja gcc/11.2.0
module load oneapi/eng-compiler/2022.12.30.002
module use /lus/gila/projects/Aurora_deployment/intel_anl_shared_CNDA/intel-gpu-umd/modulefiles/intel_compute_runtime/release
module load 2023.01.25-5b66ab0
module unload intel_compute_runtime/release/agama-devel-524
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
export SYCL_CACHE_PERSISTENT=1
unset SYCL_DEVICE_FILTER
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
mkdir -p cl_cache

cmake -DYAKL_ARCH="SYCL" \
      -DYAKL_SYCL_FLAGS="-O3" \
      -DCMAKE_CXX_FLAGS="-O3 -fsycl -sycl-std=2020 -fsycl-unnamed-lambda -fsycl-device-code-split=per_kernel -fsycl-targets=spir64_gen -Xsycl-target-backend \"-device 12.60.7\"" \
      -DYAKL_F90_FLAGS="-O3" \
      ../../..
