#!/bin/bash

cd /home/imn/yakl_ctest/scripts

source /home/imn/.profile

curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-serial-debug.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-serial-debug-nogator.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-openmp-opt.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/ctest_script.cmake
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-cuda-debug.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/run_all.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-cuda-opt.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-serial-opt.sh

chmod 744 *.sh

./run_all.sh

