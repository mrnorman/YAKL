#!/bin/bash

./cmakeclean.sh

unset GATOR_DISABLE

export CC=gcc
export CXX=g++
unset CXXFLAGS
export FFLAGS="-O0 -g"

cmake -DYAKL_CXX_FLAGS="-O0 -g -DYAKL_DEBUG" ..

