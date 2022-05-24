#!/bin/bash
#BSUB -P stf006
#BSUB -W 1:00
#BSUB -nnodes 1
#BSUB -J yakl_summit_unit
#BSUB -o yakl_summit_unit.%J
#BSUB -e yakl_summit_unit.%J

cd /ccs/home/imn/yakl_ctest/scripts/summit

source /ccs/home/imn/.bash_profile

curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/summit/ctest_script.cmake
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/summit/main-cuda-gnu-opt.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/summit/main-openmp45-ibm-opt.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/summit/main-serial-gnu-debug-nogator.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/summit/main-serial-gnu-debug.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/summit/main-serial-gnu-opt.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/summit/main-serial-ibm-opt.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/summit/main-serial-llvm-opt.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/summit/main-serial-nvhpc-opt.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/summit/run_all.sh

chmod 744 *.sh

./run_all.sh

