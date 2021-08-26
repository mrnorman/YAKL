#!/bin/bash

source $MODULESHOME/init/bash
module rm rocm craype-accel-amd-gfx908
module load PrgEnv-gnu craype-x86-rome

export CTEST_BUILD_NAME=master-serial-gnu-debug

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

test_home=/gpfs/alpine/stf006/scratch/imn/yakl_ctest/spock

export YAKL_CTEST_SRC=${test_home}/../YAKL
export YAKL_CTEST_BIN=${test_home}/scratch
export CTEST_YAKL_ARCH=""
export CTEST_CXX_FLAGS="-O0 -fprofile-arcs -ftest-coverage -g -DYAKL_DEBUG"
export CTEST_C_FLAGS="-O0 -fprofile-arcs -ftest-coverage -g"
export CTEST_F90_FLAGS="-O0 -fprofile-arcs -ftest-coverage -g"
export CTEST_LD_FLAGS="-fprofile-arcs -ftest-coverage"
export CTEST_GCOV=1
export CTEST_VALGRIND=0
export CTEST_MPI_COMMAND=""

ctest_dir=`pwd`
cd ${YAKL_CTEST_SRC}
git fetch origin
git checkout master
git reset --hard origin/master
git submodule update --init --recursive

rm -rf /gpfs/alpine/stf006/scratch/imn/yakl_ctest/spock/scratch/*

cd ${ctest_dir}

ctest -S ctest_script.cmake

