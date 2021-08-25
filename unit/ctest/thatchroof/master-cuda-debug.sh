#!/bin/bash

export CTEST_BUILD_NAME=master-cuda-gnu-debug

unset GATOR_DISABLE
unset OMP_NUM_THREADS
unset CXXFLAGS
unset FFLAGS
unset CFLAGS
unset FCLAGS
unset CUDAARCHS
unset CUDACXX
unset CUDAFLAGS
unset CUDAHOSTCXX
unset HIPCXX
unset HIPFLAGS

export CC=gcc
export CXX=g++
export FC=gfortran

test_home=/home/imn/yakl_ctest

export YAKL_CTEST_SRC=${test_home}/YAKL
export YAKL_CTEST_BIN=${test_home}/scratch
export CTEST_YAKL_ARCH="CUDA"
export CTEST_CUDA_FLAGS="-O0 -g -G -arch sm_35 -ccbin g++ -DTHRUST_IGNORE_CUB_VERSION_CHECK -DYAKL_DEBUG"
export CTEST_C_FLAGS="-O0 -g"
export CTEST_F90_FLAGS="-O0 -g"
export CTEST_LD_FLAGS=""
export CTEST_GCOV=0
export CTEST_VALGRIND=0

ctest_dir=`pwd`
cd ${YAKL_CTEST_SRC}
git fetch origin
git checkout master
git reset --hard origin/master
git submodule update --init --recursive

rm -rf /home/imn/yakl_ctest/scratch/*

cd ${ctest_dir}

ctest -S ctest_script.cmake

