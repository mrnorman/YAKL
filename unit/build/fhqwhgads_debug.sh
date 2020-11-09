#!/bin/bash

./cmakeclean.sh

unset GATOR_DISABLE

export CC=gcc
export CXX=g++
export CXXFLAGS="-O0 -g -DYAKL_DEBUG"
export FFLAGS="-O0 -g"

cmake ..

