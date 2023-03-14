#!/bin/bash
module purge
module load intel/oneapi
module load intel/mpi-utils
module load intel-comp-rt/ci-neo-master
#module load intel-comp-rt
module load cmake/3.24.2

export CC=mpiicx
export CXX=mpiicpx
export FC=mpiifx

export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=0

export aot_flags="-fsycl-targets=spir64_gen -fsycl-force-target=spir64_gen -Xs \"-device 0x0bd5 -revision_id 0x2f\""

export ZE_AFFINITY_MASK=0.0
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
