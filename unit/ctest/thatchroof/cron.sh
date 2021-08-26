#!/bin/bash

cd /home/imn/yakl_ctest/scripts

source /home/imn/.profile

curl -O https://raw.githubusercontent.com/mrnorman/YAKL/master/unit/ctest/thatchroof/master-serial-debug.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/master/unit/ctest/thatchroof/master-serial-debug-nogator.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/master/unit/ctest/thatchroof/master-openmp-opt.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/master/unit/ctest/thatchroof/ctest_script.cmake
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/master/unit/ctest/thatchroof/master-cuda-debug.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/master/unit/ctest/thatchroof/run_all.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/master/unit/ctest/thatchroof/master-cuda-opt.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/master/unit/ctest/thatchroof/master-serial-opt.sh

chmod 744 *.sh

./run_all.sh

