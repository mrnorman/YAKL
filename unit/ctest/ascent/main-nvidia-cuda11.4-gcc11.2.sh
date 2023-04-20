#!/bin/bash
source reset_env.sh
module load DefApps gcc/11.2.0 cuda/11.4.2 cmake

###############################################
## User configurable options
###############################################
export CTEST_BUILD_NAME=main-nvidia-cuda11.4-gcc11.2
export CC=gcc
export CXX=g++
export FC=gfortran
export YAKL_ARCH="CUDA"
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
export YAKL_OPENMP_FLAGS=""
export YAKL_CUDA_FLAGS="-arch sm_70 -O3 --use_fast_math -ccbin g++"
export YAKL_HIP_FLAGS=""
export YAKL_SYCL_FLAGS=""
export CTEST_GCOV=0
export CTEST_VALGRIND=0
export MPI_COMMAND="jsrun -n 1 -a 1 -c 1 -g 1"
# export GATOR_DISABLE=0
# export GATOR_INITIAL_MB=1024
# export GATOR_GROW_MB=1024
# export GATOR_BLOCK_BYTES=1024
###############################################

test_home=/gpfs/wolf/cli115/proj-shared/yakl-testing
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
ctest -j 8 -S ctest_script.cmake
