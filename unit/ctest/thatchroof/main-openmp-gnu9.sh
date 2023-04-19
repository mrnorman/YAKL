#!/bin/bash
source reset_env.sh


###############################################
## User configurable options
###############################################
export CTEST_BUILD_NAME=main-openmp-gnu9
export CC=gcc-9
export CXX=g++-9
export FC=gfortran-9
export YAKL_ARCH="OPENMP"
export YAKL_VERBOSE=OFF
export YAKL_VERBOSE_FILE=OFF
export YAKL_DEBUG=OFF
export YAKL_HAVE_MPI=OFF
export YAKL_ENABLE_STREAMS=ON
export YAKL_AUTO_PROFILE=OFF
export YAKL_PROFILE=ON
export YAKL_AUTO_FENCE=OFF
export YAKL_B4B=OFF
export YAKL_MANAGED_MEMORY=OFF
export YAKL_MEMORY_DEBUG=OFF
export YAKL_TARGET_SUFFIX=""
export YAKL_F90_FLAGS="-O3"
export YAKL_CXX_FLAGS=""
export YAKL_OPENMP_FLAGS="-O3 -fopenmp"
export YAKL_CUDA_FLAGS=""
export YAKL_HIP_FLAGS=""
export YAKL_SYCL_FLAGS=""
export CTEST_GCOV=0
export CTEST_VALGRIND=0
# export GATOR_DISABLE=0
# export GATOR_INITIAL_MB=1024
# export GATOR_GROW_MB=1024
# export GATOR_BLOCK_BYTES=1024
###############################################

test_home=/home/imn/yakl_ctest
ctest_dir=`pwd`
export YAKL_CTEST_SRC=${test_home}/YAKL
export YAKL_CTEST_BIN=${test_home}/scratch
mkdir -p $YAKL_CTEST_BIN
rm -rf ${YAKL_CTEST_BIN}/*
cd $test_home
[ ! -d "${YAKL_CTEST_SRC}" ] && git clone git@github.com:mrnorman/YAKL.git
cd ${YAKL_CTEST_SRC}
git fetch origin
git checkout main
git reset --hard origin/main
cd ${ctest_dir}
ctest -S ctest_script.cmake

