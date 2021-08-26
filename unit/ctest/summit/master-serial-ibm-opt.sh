#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load DefApps xl cmake

export CTEST_BUILD_NAME=master-serial-ibm-opt

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

export CC=xlc_r
export CXX=xlc++_r
export FC=xlf90_r

test_home=/gpfs/alpine/stf006/scratch/imn/yakl_ctest/summit

export YAKL_CTEST_SRC=${test_home}/../YAKL
export YAKL_CTEST_BIN=${test_home}/scratch
export CTEST_YAKL_ARCH=""
export CTEST_CXX_FLAGS="-O3"
export CTEST_C_FLAGS="-O3"
export CTEST_F90_FLAGS="-O3"
export CTEST_LD_FLAGS=""
export CTEST_GCOV=0
export CTEST_VALGRIND=0
export CTEST_MPI_COMMAND="jsrun -n 1 -a 1 -c 1 -g 1"

ctest_dir=`pwd`
cd ${YAKL_CTEST_SRC}
git fetch origin
git checkout master
git reset --hard origin/master
git submodule update --init --recursive

rm -rf /gpfs/alpine/stf006/scratch/imn/yakl_ctest/summit/scratch/*

cd ${ctest_dir}

ctest -S ctest_script.cmake

