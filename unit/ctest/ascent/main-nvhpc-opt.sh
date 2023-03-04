#!/bin/bash

test_home=/gpfs/wolf/cli115/proj-shared/yakl-testing
export YAKL_CTEST_SRC=${test_home}/YAKL
export YAKL_CTEST_BIN=${test_home}/scratch
rm -rf ${YAKL_CTEST_SRC}
rm -rf ${YAKL_CTEST_BIN}

source $MODULESHOME/init/bash
module purge
module load DefApps nvhpc cuda cmake

export CTEST_BUILD_NAME=main-nvhpc-opt
export CC=mpicc
export CXX=mpic++
export FC=mpif90

# export CTEST_YAKL_ARCH="CUDA"
export CTEST_CXX_FLAGS="-O3"
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

