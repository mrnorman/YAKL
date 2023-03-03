#!/bin/bash

test_home=$HOME/yakl-testing
mkdir -p ${test_home}
export YAKL_CTEST_SRC=${test_home}/YAKL
export YAKL_CTEST_BIN=${test_home}/scratch
rm -rf ${YAKL_CTEST_SRC}
rm -rf ${YAKL_CTEST_BIN}
export CTEST_BUILD_NAME=main-intel-opt

module purge
module restore
module load spack
module load cmake 
export CC=mpicc
export CXX=mpicxx
export FC=mpifort

export CTEST_YAKL_ARCH="SYCL"
#export CTEST_SYCL_FLAGS="-O3 -DYAKL_ENABLE_STREAMS -fsycl -sycl-std=2020 -fsycl-unnamed-lambda -I/soft/restricted/CNDA/updates/2022.12.30.001/oneapi/compiler/trunk-20230201/compiler/linux/include/sycl"
export CTEST_SYCL_FLAGS="-O3 -DYAKL_ENABLE_STREAMS -fsycl -sycl-std=2020 -fsycl-unnamed-lambda -I/soft/restricted/CNDA/updates/2022.12.30.001/oneapi/compiler/trunk-20230201/compiler/linux/include/sycl \
	-fsycl-default-sub-group-size=16 -fsycl-device-code-split=per_kernel -fsycl-targets=spir64_gen "
# -Xsycl-target-backend \' -device 12.60.7 \' "
# export CTEST_SYCL_FLAGS="-O3 -DYAKL_ENABLE_STREAMS"
export CTEST_CXX_FLAGS="-O3 -DYAKL_ENABLE_STREAMS -fsycl -sycl-std=2020 -fsycl-unnamed-lambda -I/soft/restricted/CNDA/updates/2022.12.30.001/oneapi/compiler/trunk-20230201/compiler/linux/include/sycl \
	-fsycl-default-sub-group-size=16 -fsycl-device-code-split=per_kernel -fsycl-targets=spir64_gen " 
# -Xsycl-target-backend \' -device 12.60.7 \' "
export CTEST_F90_FLAGS="-O3"
export CTEST_LD_FLAGS=""
export CTEST_GCOV=0
export CTEST_VALGRIND=0

ctest_dir=`pwd`
cd ${test_home}
git clone --recurse-submodules git@github.com:mrnorman/YAKL.git

cd ${ctest_dir}

ctest -j 4 -S ctest_script.cmake


