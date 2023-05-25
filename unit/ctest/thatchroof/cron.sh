#!/bin/bash
source /etc/profile
source /home/imn/.profile

mkdir -p /home/imn/yakl_ctest/scripts
cd /home/imn/yakl_ctest/scripts

curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/ctest_script.cmake
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-cpu-gnu11-debug.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-cpu-gnu11-debug-valgrind.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-nvidia-cuda11.1-gcc11.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-nvidia-cuda11.2-gcc11.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-nvidia-cuda11.3-gcc11.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-nvidia-cuda11.4-gcc11.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-nvidia-cuda11.5-gcc11.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-nvidia-cuda11.6-gcc11.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-nvidia-cuda11.7-gcc11.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-nvidia-cuda11.8-gcc11.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-nvidia-cuda12.0-gcc11-managed.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-nvidia-cuda12.0-gcc11.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-nvidia-cuda12-gcc11-debug.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-nvidia-nvhpc23.3.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-openmp-aocc3.2.0.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-openmp-aocc4.0.0.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-openmp-gnu10.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-openmp-gnu11.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-openmp-gnu12.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-openmp-gnu8.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-openmp-gnu9.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-openmp-intel2021.6.0.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-openmp-llvm14.0.6
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-openmp-llvm16.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/main-openmp-nvhpc23.3.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/reset_env.sh
curl -O https://raw.githubusercontent.com/mrnorman/YAKL/main/unit/ctest/thatchroof/run_all.sh

chmod 744 *.sh

./run_all.sh

