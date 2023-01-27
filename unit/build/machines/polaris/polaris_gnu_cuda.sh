#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load PrgEnv-gnu
module load craype-x86-milan
module load cmake
module load cpe-cuda
module load cudatoolkit-standalone
module list

../../cmakeclean.sh

unset GATOR_DISABLE

export CC=cc
export CXX=CC
export FC=ftn
unset CXXFLAGS
unset FFLAGS

export CUDA_VISIBLE_DEVICES=0

cmake -DYAKL_ARCH="CUDA"                                \
      -DYAKL_CUDA_FLAGS="-O3 -DYAKL_PROFILE --use_fast_math -arch sm_80 -ccbin CC -DTHRUST_IGNORE_CUB_VERSION_CHECK" \
      -DYAKL_F90_FLAGS="-O3"                            \
      -DMPI_COMMAND="mpiexec -n 1 " \
      ../../..
