#!/bin/bash

./main-cuda-debug.sh
./main-cuda-opt.sh
./main-openmp-opt.sh
./main-serial-debug-nogator.sh
./main-serial-debug.sh
./main-serial-opt.sh

