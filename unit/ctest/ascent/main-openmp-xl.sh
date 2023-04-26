#!/bin/bash
source reset_env.sh
module load DefApps xl/16.1.1-10 cmake

test_home=/gpfs/wolf/cli115/proj-shared/yakl-testing

###############################################
## User configurable options
###############################################
export CTEST_BUILD_NAME=main-openmp-xl16.1.1-10
export CC=mpicc
export CXX=mpic++
export FC=mpif90
export YAKL_ARCH="OPENMP"
export YAKL_VERBOSE=OFF
export YAKL_VERBOSE_FILE=OFF
export YAKL_DEBUG=OFF
export YAKL_HAVE_MPI=OFF
export YAKL_ENABLE_STREAMS=OFF
export YAKL_AUTO_PROFILE=OFF
export YAKL_PROFILE=ON
export YAKL_AUTO_FENCE=OFF
export YAKL_B4B=OFF
export YAKL_MANAGED_MEMORY=OFF
export YAKL_MEMORY_DEBUG=OFF
export YAKL_TARGET_SUFFIX=""
export YAKL_F90_FLAGS="-O3"
export YAKL_CXX_FLAGS=""
export YAKL_OPENMP_FLAGS="-O3 -qsmp=omp"
export YAKL_CUDA_FLAGS=""
export YAKL_HIP_FLAGS=""
export YAKL_SYCL_FLAGS=""
export CTEST_GCOV=0
export CTEST_VALGRIND=0
export MPI_COMMAND="jsrun -n 1 -a 1 -c 42"
export OMP_NUM_THREADS=42
# export GATOR_DISABLE=0
# export GATOR_INITIAL_MB=1024
# export GATOR_GROW_MB=1024
# export GATOR_BLOCK_BYTES=1024
###############################################

ctest_dir=`pwd`
export YAKL_CTEST_SRC=${test_home}/YAKL
export YAKL_CTEST_BIN=${test_home}/scratch
rm -rf ${YAKL_CTEST_BIN}
mkdir -p $YAKL_CTEST_BIN
cd ${YAKL_CTEST_SRC}
git reset --hard origin/main
cd ${ctest_dir}
ctest -j 4 -S ctest_script.cmake

