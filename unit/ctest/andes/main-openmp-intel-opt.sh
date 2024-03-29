#!/bin/bash

source /sw/andes/lmod/lmod/init/bash
module load intel cmake

export CTEST_BUILD_NAME=main-openmp-intel-opt

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

export OMP_NUM_THREADS=32

export CC=icc
export CXX=icpc
export FC=ifort

test_home=/gpfs/alpine/stf006/scratch/imn/yakl_ctest/andes

export YAKL_CTEST_SRC=${test_home}/../YAKL
export YAKL_CTEST_BIN=${test_home}/scratch
export CTEST_YAKL_ARCH="OPENMP"
export CTEST_OPENMP_FLAGS="-O3 -qopenmp"
export CTEST_C_FLAGS="-O3"
export CTEST_F90_FLAGS="-O3"
export CTEST_LD_FLAGS=""
export CTEST_GCOV=0
export CTEST_VALGRIND=0
export CTEST_MPI_COMMAND=""

ctest_dir=`pwd`
cd ${YAKL_CTEST_SRC}
git fetch origin
git checkout main
git reset --hard origin/main
git submodule update --init --recursive

rm -rf /gpfs/alpine/stf006/scratch/imn/yakl_ctest/andes/scratch/*

cd ${ctest_dir}

ctest -S ctest_script.cmake

