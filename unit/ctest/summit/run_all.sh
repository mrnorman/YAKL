#!/bin/bash

./master-cuda-gnu-opt.sh
./master-openmp45-ibm-opt.sh
./master-serial-gnu-debug-nogator.sh
./master-serial-gnu-debug.sh
./master-serial-gnu-opt.sh
./master-serial-ibm-opt.sh
./master-serial-llvm-opt.sh
./master-serial-nvhpc-opt.sh

