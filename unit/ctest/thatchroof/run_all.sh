#!/bin/bash

./master-cuda-debug.sh
./master-cuda-opt.sh
./master-openmp-opt.sh
./master-serial-debug-nogator.sh
./master-serial-debug.sh
./master-serial-opt.sh

