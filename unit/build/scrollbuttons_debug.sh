#!/bin/bash

./cmakeclean.sh

export CXXFLAGS="-O0 -g -DYAKL_DEBUG"

cmake ..

