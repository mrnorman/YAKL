# Running the YAKL unit tests

## Initialize the Kokkos submodules

YAKL pins Kokkos 4.7.00 and 5.2.0 as submodules under `external/`. Their URLs use SSH, so your GitHub SSH key must be configured.
The submodules have `update = none` to prevent a generic recursive submodule update from changing them implicitly. Initialize them
explicitly with `--checkout`:

```bash
git clone git@github.com:mrnorman/YAKL.git
cd YAKL
git submodule update --init --checkout external/kokkos-4.7.00 external/kokkos-5.2.0
```

## Run the two-version matrix

Source one machine environment, then run the matrix driver from `unit/build`:

```bash
cd unit/build
source machines/[machine_name]/machine_option.env
./test_kokkos_versions.sh
```

The driver configures, builds, and runs `ctest -V` for the complete test suite against each version. It uses independent build
trees at `unit/build/kokkos-matrix/4.7.00` and `unit/build/kokkos-matrix/5.2.0`, continues to the second version if the first fails,
and returns a failure status if either version fails. To limit build parallelism, for example:

```bash
YAKL_BUILD_JOBS=8 ./test_kokkos_versions.sh
```

### Run every thatchroof environment

On thatchroof, one driver runs every `machines/thatchroof/thatchroof_*.env` environment against both supported Kokkos
submodules:

```bash
cd unit/build
./test_thatchroof_matrix.sh
```

The environments are discovered automatically and run sequentially. Currently this tests CUDA debug, OpenMP, and Threads
with Kokkos 4.7.00 and 5.2.0, for six configure/build/test combinations. Each combination has an independent build tree under
`unit/build/kokkos-matrix/thatchroof/<environment>/<version>`. The driver continues after failures and exits unsuccessfully
after the complete matrix if any environment failed. `YAKL_BUILD_JOBS` limits build parallelism just as it does for the
single-environment driver. To place the matrix builds elsewhere, set `YAKL_THATCHROOF_MATRIX_BUILD_ROOT`:

```bash
YAKL_BUILD_JOBS=8 \
YAKL_THATCHROOF_MATRIX_BUILD_ROOT=/tmp/yakl-thatchroof-matrix \
./test_thatchroof_matrix.sh
```

### Run every Frontier environment

On a Frontier compute node, the corresponding driver runs every `machines/frontier/frontier_*.env` environment against both
Kokkos submodules:

```bash
cd unit/build
./test_frontier_matrix.sh
```

This currently runs optimized HIP, debug HIP, CPU/Serial, and OpenMP environments with Kokkos 4.7.00 and 5.2.0, for eight
sequential configure/build/test combinations. Builds are isolated under
`unit/build/kokkos-matrix/frontier/<environment>/<version>`. Use `YAKL_BUILD_JOBS` to limit build parallelism or
`YAKL_FRONTIER_MATRIX_BUILD_ROOT` to relocate the build trees:

```bash
YAKL_BUILD_JOBS=16 \
YAKL_FRONTIER_MATRIX_BUILD_ROOT="$SCRATCH/yakl-frontier-matrix" \
./test_frontier_matrix.sh
```

For CUDA builds, the matrix can select a different toolkit for each Kokkos release through
`YAKL_KOKKOS_4_7_CUDA_ROOT` and `YAKL_KOKKOS_5_2_CUDA_ROOT`. On thatchroof, the GPU-debug environment uses CUDA 12.2 for
Kokkos 4.7 and CUDA 13.3 for Kokkos 5.2. Kokkos 4.7 does not compile against CUDA 13 because CUDA 13 changed APIs used by that
release; Kokkos 5.2's C++20 `mdspan` implementation does not compile under NVCC 12.2.

YAKL supports Kokkos 4.7.00 through the current Kokkos release. Kokkos 4.7.00 is the oldest tested version because it contains
the CUDA `RangePolicy` fixes required for iteration counts above `UINT_MAX`.

## Run one Kokkos version

`cmakescript.sh` defaults to the Kokkos 5.2.0 submodule. Override `Kokkos_HOME` to configure the current build directory against
one specific source tree:

```bash
cd unit/build
source machines/thatchroof/thatchroof_gpu_debug.env
CUDAToolkit_ROOT="$YAKL_KOKKOS_4_7_CUDA_ROOT" \
CUDA_ROOT="$YAKL_KOKKOS_4_7_CUDA_ROOT" \
PATH="$YAKL_KOKKOS_4_7_CUDA_ROOT/bin:$PATH" \
Kokkos_HOME="$PWD/../../external/kokkos-4.7.00" \
./cmakescript.sh
cmake --build . --parallel 8
ctest -V
```

When selecting a CUDA installation outside a machine environment, set it before configuring:

```bash
export CUDAToolkit_ROOT=/usr/local/cuda-12.2
export CUDA_ROOT="$CUDAToolkit_ROOT"
export PATH="$CUDAToolkit_ROOT/bin:$PATH"
```
