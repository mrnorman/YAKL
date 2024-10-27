#!/bin/bash
source reset_env.sh


###############################################
## User configurable options
###############################################
export CTEST_BUILD_NAME=main-cpu-gnu11-debug-valgrind
export CC=gcc-11
export CXX=g++-11
export FC=gfortran-11
export YAKL_ARCH=""
export YAKL_VERBOSE=OFF
export YAKL_VERBOSE_FILE=OFF
export YAKL_DEBUG=ON
export YAKL_HAVE_MPI=OFF
export YAKL_ENABLE_STREAMS=OFF
export YAKL_AUTO_PROFILE=OFF
export YAKL_PROFILE=OFF
export YAKL_AUTO_FENCE=ON
export YAKL_B4B=OFF
export YAKL_MANAGED_MEMORY=OFF
export YAKL_MEMORY_DEBUG=OFF
export YAKL_TARGET_SUFFIX="debug"
export YAKL_F90_FLAGS="-O0 -g"
export YAKL_CXX_FLAGS="-O0 -g"
export YAKL_OPENMP_FLAGS=""
export YAKL_CUDA_FLAGS=""
export YAKL_HIP_FLAGS=""
export YAKL_SYCL_FLAGS=""
export CTEST_GCOV=0
export CTEST_VALGRIND=1
export GATOR_DISABLE=1  # Disable pool for valgrind mem access check accuracy
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

