#!/bin/bash

module purge
module load oneapi/eng-compiler/2023.10.15.002
module load spack cmake


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

${SCRIPT_DIR}/../../cmakeclean.sh

unset GATOR_DISABLE

export CC=mpicc
export CXX=mpicxx
export FC=mpifort
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="SYCL" \
      -DYAKL_SYCL_FLAGS="-fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend \"-device pvc\"" \
      -DCMAKE_CXX_FLAGS="-O3" \
      -DYAKL_F90_FLAGS="-O3" \
      -DYAKL_C_FLAGS="-O3"   \
      ${SCRIPT_DIR}/../../..
