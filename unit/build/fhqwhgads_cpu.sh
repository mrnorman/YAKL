#!/bin/bash

./cmakeclean.sh

unset GATOR_DISABLE

export CC=gcc
export CXX=g++
unset CXXFLAGS
export FFLAGS="-O3"

cmake -DYAKL_CXX_FLAGS="-O3" ..

