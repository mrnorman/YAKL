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

export ZE_AFFINITY_MASK=0.0

cmake -DYAKL_ARCH="SYCL" \
      -DYAKL_SYCL_FLAGS="-O3" \
      -DCMAKE_CXX_FLAGS="-O3 -fsycl -sycl-std=2020 -fsycl-unnamed-lambda -fsycl-default-sub-group-size=16 -fsycl-device-code-split=per_kernel -fsycl-targets=spir64_gen -Xsycl-target-backend \"-device 12.60.7\"" \
      -DYAKL_F90_FLAGS="-O3" \
      -DYAKL_C_FLAGS="-O3"   \
      ../../..
