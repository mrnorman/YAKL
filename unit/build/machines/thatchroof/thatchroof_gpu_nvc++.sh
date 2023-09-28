#!/bin/bash

source /usr/share/modules/init/bash
module purge
module load cmake-3.23.2-gcc-11.1.0-kvgnqc6 nvhpc-23.3-gcc-11.1.0-lyprlux netcdf-c-4.9.2-nvhpc-23.3-l7riqnm

../../cmakeclean.sh

unset GATOR_DISABLE

export CC=nvc
export CXX=nvc++
export FC=nvfortran
unset CXXFLAGS
unset FFLAGS

cmake -DYAKL_ARCH="CUDA"                                                                               \
      -DYAKL_CUDA_FLAGS="-O3 -DYAKL_ENABLE_STREAMS -gpu=cc86 -traceback -I`nc-config --includedir`" \
      -DYAKL_F90_FLAGS="-O3"                                                                           \
      -DCMAKE_INSTALL_PREFIX="`pwd`" \
      -DNETCDF_LINK_FLAGS="`nc-config --libs`"        \
      ../../..

