#!/bin/bash

./main-cuda-gnu-opt.sh
./main-openmp45-ibm-opt.sh
./main-serial-gnu-debug-nogator.sh
./main-serial-gnu-debug.sh
./main-serial-gnu-opt.sh
./main-serial-ibm-opt.sh
./main-serial-llvm-opt.sh
./main-serial-nvhpc-opt.sh

