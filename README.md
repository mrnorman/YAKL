# YAKL: YAKL is A Kokkos Layer

YAKL is a C++20 layer over [Kokkos](https://github.com/kokkos/kokkos) for performance-portable scientific applications and
Fortran-to-C++ code porting. It supports Kokkos 4.7.00 through the current Kokkos release.

YAKL provides:

- `Array` and `Array_F`, dynamic multidimensional arrays with C-style and Fortran-style indexing;
- `SArray` and `SArray_F`, fixed-size multidimensional arrays for host and device code;
- `parallel_for` and `parallel_for_F`, including general bounds, runtime tiling, and optional autotuning;
- `DeviceSpace`, a Kokkos memory space backed by YAKL's optional device memory pool;
- host/device intrinsics, componentwise array operations, random-number generation, and timers; and
- optional NetCDF, PNetCDF, and Fortran allocation interfaces.

## Build and use YAKL

YAKL is intended to be included in an application's CMake build with `add_subdirectory`. Configure the desired Kokkos
backend and architecture before adding Kokkos, then add YAKL and link its `yakl` target:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_application LANGUAGES C CXX Fortran)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the Kokkos backend and architecture appropriate for this build.
# Examples:
# set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "")
# set(Kokkos_ENABLE_CUDA ON CACHE BOOL "")
# set(Kokkos_ARCH_AMPERE86 ON CACHE BOOL "")

add_subdirectory(external/kokkos)
add_subdirectory(external/YAKL)

add_executable(my_application main.cpp)
target_link_libraries(my_application PRIVATE yakl)
```

`Kokkos::kokkos` is a public dependency of `yakl`, so applications normally only need to link `yakl`. Backend-specific
compiler selection and Kokkos options remain the responsibility of the enclosing application.

Include the main API with:

```cpp
#include "YAKL.h"
```

The application owns the Kokkos runtime. Initialize Kokkos before YAKL, destroy all YAKL device allocations before
`yakl::finalize()`, and finalize Kokkos last:

```cpp
int main(int argc, char **argv) {
  Kokkos::initialize(argc,argv);
  {
    yakl::init();
    {
      yakl::Array<double *> values("values",1024);
      yakl::parallel_for("initialize",values.size(),KOKKOS_LAMBDA (size_t i) {
        values(i) = static_cast<double>(i);
      });
    }
    yakl::finalize();
  }
  Kokkos::finalize();
}
```

Allocation and deallocation through `yakl::DeviceSpace` must occur between `yakl::init()` and `yakl::finalize()`.

## API documentation

The current, traversable Markdown API reference starts at [`doc/README.md`](doc/README.md). It covers lifecycle and build
configuration, arrays, parallel execution, algorithms, memory management, random numbers, timers, and optional file I/O.

## Cite YAKL: https://link.springer.com/article/10.1007/s10766-022-00739-0

Primary Developer: Matt Norman (Oak Ridge National Laboratory) - mrnorman.github.io
