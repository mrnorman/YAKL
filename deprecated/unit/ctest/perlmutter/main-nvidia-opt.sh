#!/bin/bash

# source $MODULESHOME/init/bash
module reset
module load cmake PrgEnv-nvidia cudatoolkit craype-accel-nvidia80

# test_home=${PSCRATCH}/yakl-testing
test_home=/global/cfs/cdirs/e3sm/yakl-testing
mkdir -p ${test_home}
rm -rf ${test_home}/YAKL
rm -rf ${test_home}/scratch

export CTEST_BUILD_NAME=main-nvidia-opt

export CC=cc
export CXX=CC
export FC=ftn

export YAKL_CTEST_SRC=${test_home}/YAKL
export YAKL_CTEST_BIN=${test_home}/scratch
export CTEST_YAKL_ARCH="CUDA"
export CTEST_CUDA_FLAGS="-O3 --use_fast_math -arch sm_80 -ccbin CC -DTHRUST_IGNORE_CUB_VERSION_CHECK"
export CTEST_C_FLAGS="-O3"
export CTEST_F90_FLAGS="-O3"
export CTEST_LD_FLAGS=""
export CTEST_GCOV=0
export CTEST_VALGRIND=0
export CTEST_MPI_COMMAND=""

ctest_dir=`pwd`
cd ${test_home}
git clone --recurse-submodules git@github.com:mrnorman/YAKL.git

cd ${ctest_dir}

ctest -j 4 -S ctest_script.cmake

