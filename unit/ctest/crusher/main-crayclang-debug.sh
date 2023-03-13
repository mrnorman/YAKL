#!/bin/bash

test_home=/gpfs/alpine/cli133/proj-shared/yakl-testing/
export CTEST_BUILD_NAME=main-crayclang-debug

source $MODULESHOME/init/bash
module reset
module load PrgEnv-cray craype-accel-amd-gfx90a rocm
unset GATOR_DISABLE
unset OMP_NUM_THREADS
export CC=cc
export CXX=CC
export FC=ftn

export YAKL_CTEST_SRC=${test_home}/YAKL
export YAKL_CTEST_BIN=${test_home}/scratch
export CTEST_YAKL_ARCH="HIP"
export CTEST_HIP_FLAGS="-O0 -g -DYAKL_DEBUG -DYAKL_ENABLE_STREAMS -Wno-tautological-pointer-compare -Wno-unused-result -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=${ROCM_PATH} --offload-arch=gfx90a -x hip"
export CTEST_C_FLAGS="-O0 -g -DYAKL_DEBUG"
export CTEST_F90_FLAGS="-O0 -g -L${ROCM_PATH}/lib -lamdhip64"
export CTEST_LD_FLAGS="--rocm-path=${ROCM_PATH} -L${ROCM_PATH}/lib -lamdhip64"
export CTEST_GCOV=0
export CTEST_VALGRIND=0

ctest_dir=`pwd`
cd ${YAKL_CTEST_SRC}
git reset --hard origin/main
git submodule update --init --recursive

rm -rf ${YAKL_CTEST_BIN}
mkdir ${YAKL_CTEST_BIN}

cd ${ctest_dir}

ctest -S ctest_script.cmake
