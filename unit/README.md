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
