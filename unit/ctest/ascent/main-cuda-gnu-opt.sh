#!/bin/bash

test_home=/gpfs/wolf/cli115/proj-shared/yakl-testing
export YAKL_CTEST_SRC=${test_home}/YAKL
export YAKL_CTEST_BIN=${test_home}/scratch
rm -rf ${YAKL_CTEST_SRC}
rm -rf ${YAKL_CTEST_BIN}

source $MODULESHOME/init/bash
module purge
module load DefApps gcc/11.2.0 cuda/11.4.2 cmake

export CTEST_BUILD_NAME=main-cuda-gnu-opt
export CC=gcc
export CXX=g++
export FC=gfortran


export CTEST_YAKL_ARCH="CUDA"
export CTEST_CUDA_FLAGS="-O3 --use_fast_math -arch sm_70 -ccbin g++ -DTHRUST_IGNORE_CUB_VERSION_CHECK"
export CTEST_C_FLAGS="-O3"
export CTEST_F90_FLAGS="-O3"
export CTEST_LD_FLAGS=""
export CTEST_GCOV=0
export CTEST_VALGRIND=0
export CTEST_MPI_COMMAND="jsrun -n 1 -a 1 -c 1 -g 1"

ctest_dir=`pwd`
# cd ${test_home}
# git clone --recurse-submodules git@github.com:mrnorman/YAKL.git

cd ${ctest_dir}

ctest -j 4 -S ctest_script.cmake

