#!/bin/bash

module purge
module use /soft/restricted/CNDA/modulefiles
module load oneapi
module use /soft/modulefiles
module load cmake/3.23.2
module load bbfft

../../cmakeclean.sh

unset GATOR_DISABLE

export CC=icx
export CXX=icpx
export FC=ifx
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="SYCL" \
      -DYAKL_SYCL_FLAGS="-fsycl -sycl-std=2020 -fsycl-unnamed-lambda -fsycl-default-sub-group-size=16 -fsycl-device-code-split=per_kernel -fsycl-targets=spir64_gen -Xsycl-target-backend \"-device 0x0bd5 -revision_id 0x2f\"" \
      -DCMAKE_CXX_FLAGS="-O3" \
      -DYAKL_F90_FLAGS="-O3" \
      -DYAKL_C_FLAGS="-O3"   \
      -DYAKL_SYCL_BBFFT=1      \
      -DYAKL_SYCL_BBFFT_AOT=1    \
      -DYAKL_SYCL_BBFFT_AOT_LEGACY_UMD=1 \
      ../../..


make -j32 FFT
