# YAKL API documentation

This directory is the Markdown API reference for YAKL. It documents public interfaces by concept instead of expanding every
rank, dimensionality, scalar type, or template specialization into a separate page.

YAKL is a C++20, header-oriented layer over Kokkos. Include the main API with:

```cpp
#include "YAKL.h"
```

The NetCDF and PNetCDF wrappers use separate extension headers and optional dependencies.

## Start here

1. [Getting started and lifecycle](getting-started.md) explains building, initialization, finalization, configuration, and a
   complete first kernel.
2. [Arrays and indexing](arrays.md) covers `Array`, `Array_F`, `SArray`, `SArray_F`, memory spaces, ownership, and transforms.
3. [Parallel execution](parallel-execution.md) covers `LoopSpec`, bounds, launch configuration, and autotuning.
4. [Intrinsics and componentwise operations](algorithms.md) describes reductions, inquiry functions, matrix helpers, and
   elementwise expressions.
5. [Random numbers and timers](random-timers.md) covers `Random`, `ScalarLiveOut`, profiling, and timer nesting.
6. [Memory management](memory.md) describes `DeviceSpace`, the pool, `LinearAllocator`, and the Fortran allocation wrappers.
7. [NetCDF and PNetCDF](io.md) documents the optional file interfaces and their collective-call rules.
8. [Compile-time configuration](configuration.md) lists user-provided macros that alter checking, profiling,
   synchronization, and MPI behavior.

## API index

| API | Purpose | Reference |
| --- | --- | --- |
| `yakl::init`, `yakl::finalize`, `yakl::InitConfig` | YAKL lifecycle and pool configuration | [Getting started](getting-started.md#lifecycle) |
| `yakl::Array`, `yakl::Array_F` | Dynamic, Kokkos-View-based multidimensional arrays | [Dynamic arrays](arrays.md#dynamic-arrays) |
| `yakl::SArray`, `yakl::SArray_F`, `yakl::Bnds` | Fixed-size inline arrays usable in host/device functions | [Static arrays](arrays.md#static-arrays) |
| `yakl::ViewType`, `yakl::COLON` | Array data-type construction and whole-dimension slicing | [Supporting types](arrays.md#supporting-types) |
| `yakl::LoopSpec`, `yakl::LoopSpec_F` | C-style and Fortran-style inclusive loop intervals | [Loop specifications](parallel-execution.md#loop-specifications) |
| `yakl::Bounds`, `yakl::Bounds_F` | Multidimensional general bounds | [Bounds](parallel-execution.md#bounds) |
| `yakl::SimpleBounds`, `yakl::SimpleBounds_F` | Multidimensional extent-only bounds | [Bounds](parallel-execution.md#bounds) |
| `yakl::Config` | Runtime tile and compile-time launch bound | [Launch configuration](parallel-execution.md#launch-configuration) |
| `yakl::parallel_for`, `yakl::parallel_for_F` | C-style and Fortran-style kernel launchers | [Launchers](parallel-execution.md#parallel_for) |
| `yakl::autotune::parallel_for[_F]` | Runtime launch-configuration search | [Autotuning](parallel-execution.md#autotuning) |
| `yakl::intrinsics::*` | Array inquiries, reductions, selection, and small matrices | [Intrinsics](algorithms.md#intrinsics) |
| `yakl::componentwise::*` | Elementwise operators and math functions | [Componentwise API](algorithms.md#componentwise-api) |
| `yakl::Random` | Allocation-free Philox random streams | [Random](random-timers.md#random) |
| `yakl::ScalarLiveOut` | A device-written scalar with explicit host transfer | [ScalarLiveOut](random-timers.md#scalarliveout) |
| `yakl::timer_*`, `yakl::Toney` | Optional nested timing and queries | [Timers](random-timers.md#timers) |
| `yakl::DeviceSpace`, `yakl::LinearAllocator` | Kokkos memory space and pool implementation | [Memory](memory.md) |
| `yakl::SimpleNetCDF` | Serial NetCDF wrapper | [NetCDF](io.md#simplenetcdf) |
| `yakl::SimplePNetCDF` | MPI/PNetCDF wrapper | [PNetCDF](io.md#simplepnetcdf) |
| Fortran `gator_mod` | Pool-backed Fortran pointer allocation | [Fortran allocation API](memory.md#fortran-allocation-api) |
| `YAKL_*`, Kokkos debug definitions | Compile-time behavior controls | [Compile-time configuration](configuration.md) |

## Common conventions

- Names ending in `_F` use Fortran conventions: one-based default indices, arbitrary signed lower bounds where supported,
  and first-index-fastest layout. Unsuffixed APIs use C conventions: zero-based indices and last-index-fastest layout.
- `yakl::DeviceSpace` arrays are normally accessed in kernels. `Kokkos::HostSpace` arrays are accessed by host code.
- Kernel launches and device copies are asynchronous unless an API explicitly promises a fence or `YAKL_AUTO_FENCE` is
  enabled. File I/O and host-result reductions necessarily complete relevant transfers before returning.
- Checks controlled by Kokkos debug macros may be absent in release builds. Preconditions remain requirements even when
  they are not checked.

[Back to the project README](../README.md)
