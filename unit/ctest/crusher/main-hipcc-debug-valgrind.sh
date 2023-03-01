#!/bin/bash

test_home=/gpfs/alpine/cli133/proj-shared/yakl-testing/
export CTEST_BUILD_NAME=main-hipcc-debug-valgrind

source $MODULESHOME/init/bash
module reset
module load PrgEnv-amd craype-accel-amd-gfx90a
unset GATOR_DISABLE
unset OMP_NUM_THREADS
export CC=cc
export CXX=hipcc
export FC=ftn

export YAKL_CTEST_SRC=${test_home}/YAKL
export YAKL_CTEST_BIN=${test_home}/scratch
export CTEST_YAKL_ARCH="HIP"
export CTEST_HIP_FLAGS="-O0 -g -DYAKL_DEBUG -DYAKL_ENABLE_STREAMS -Wno-tautological-pointer-compare -Wno-unused-result --offload-arch=gfx90a -x hip"
export CTEST_C_FLAGS="-O0 -g"
export CTEST_F90_FLAGS="-O0 -g "
export CTEST_LD_FLAGS=""
export CTEST_GCOV=0
export CTEST_VALGRIND=1

ctest_dir=`pwd`
cd ${YAKL_CTEST_SRC}
git reset --hard origin/main
git submodule update --init --recursive

rm -rf ${YAKL_CTEST_BIN}
mkdir ${YAKL_CTEST_BIN}

cd ${ctest_dir}

ctest -S ctest_script.cmake
