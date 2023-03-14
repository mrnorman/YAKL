#!/bin/bash

module purge
module use /soft/restricted/CNDA/modulefiles
module load oneapi/eng-compiler/2022.12.30.003
module use /soft/modulefiles
module load cmake/3.23.2
module load bbfft

unset GATOR_DISABLE

export CC=icx
export CXX=icpx
export FC=ifx
unset CXXFLAGS
unset FFLAGS

../../cmakeclean.sh

cmake -DYAKL_ARCH="SYCL" \
      -DYAKL_SYCL_FLAGS="-fsycl -sycl-std=2020 -fsycl-unnamed-lambda -fsycl-default-sub-group-size=16 -fsycl-device-code-split=per_kernel " \
      -DCMAKE_CXX_FLAGS="-O3" \
      -DYAKL_F90_FLAGS="-O3" \
      -DYAKL_C_FLAGS="-O3"   \
      -DYAKL_SYCL_BBFFT=1      \
      -DYAKL_SYCL_BBFFT_AOT=0    \
      ../../..


make -j32 FFT
