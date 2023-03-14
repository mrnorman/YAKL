#!/bin/bash

config="pvc"
arch="SYCL"
fft="mkl"
aot=1
bbfft_aot=1
profile_flag=""
build="Release"
unit="FFT"
legacy_umd=0

if [[ $1 == '-h' || $1 == '--help' || $1 == '-help' ]]; then
    echo "Run as follows:"
    echo "./build.sh [options]"
    echo
    echo "    -c <pvc>"
    echo "        hardware configuration environment variables to source, indicated by {config}-env.sh (default: ${config})"
    echo
    echo "    -a <SYCL>"
    echo "        backend to build for (default: ${arch})"
    echo
    echo "    -f <mkl|bbfft>"
    echo "        if using sycl, fft backend to build for (default: ${fft})"
    echo
    echo "    -u"
    echo "        unit test to build (default: ${unit})"
    echo
    echo "    -j"
    echo "        build with JIT instead of AOT (default: AOT)"
    echo
    echo "    -p"
    echo "        build with YAKL profile (default: off)"
    echo
    echo "    -l"
    echo "        build with legacy UMD (default: off)"
    echo
    echo "    -d"
    echo "        build with Debug (default: ${build})"
    echo
    exit 0
fi

# fetch input arguments, if any
while getopts "c:a:f:ujpld" flag
do
    case "${flag}" in
        c) config=${OPTARG};;
        a) arch=${OPTARG};;
        f) fft=${OPTARG};;
        u) unit=${OPTARG};;
        j) aot=0;;
        p) profile_flag="-DYAKL_PROFILE";;
        l) legacy_umd=1;;
        d) build="Debug";;
    esac
done


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source ${SCRIPT_DIR}/${config}-env.sh

echo "legacy_umd: ${legacy_umd}"
if [[ ${legacy_umd} = 1 && ${config} =~ "pvc" ]]
then
  module switch -f intel-comp-rt/ci-neo-master intel-comp-rt
fi

module list

${SCRIPT_DIR}/../../cmakeclean.sh

if [[ ${fft} = "bbfft" ]]
then
  bbfft=1
fi
if [[ ${aot} = 0 ]]
then
  bbfft_aot=0
  aot_flags=""
fi

cmake -DYAKL_ARCH="${arch}"                                                                                                                \
      -DYAKL_SYCL_FLAGS="-O3 -fsycl -sycl-std=2020 -fsycl-unnamed-lambda -fsycl-device-code-split=per_kernel ${aot_flags} ${profile_flag}" \
      -DCMAKE_CXX_FLAGS="-O3"                                                                                                              \
      -DYAKL_F90_FLAGS="-O3"                                                                                                               \
      -DYAKL_SYCL_BBFFT=${bbfft}                                                                                                           \
      -DYAKL_SYCL_BBFFT_AOT=${bbfft_aot}                                                                                                   \
      -DYAKL_SYCL_BBFFT_HOME="/nfs/site/home/omarahme/git-repos/double-batched-fft-library/install"                                        \
      -DYAKL_SYCL_BBFFT_AOT_LEGACY_UMD=${legacy_umd}                                                                                           \
      -DCMAKE_BUILD_TYPE=${build}                                                                                                          \
      ${SCRIPT_DIR}/../../..

make clean

make ${unit} -j32
