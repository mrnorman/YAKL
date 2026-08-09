# Compile-time configuration

[API home](README.md) · [Getting started](getting-started.md) · [Memory](memory.md)

This page lists preprocessor definitions a user or parent build may provide to alter YAKL behavior. Definitions should be
set consistently for YAKL and every translation unit that includes YAKL headers. With CMake, prefer target-scoped compile
definitions:

```cmake
target_compile_definitions(my_application PRIVATE YAKL_PROFILE YAKL_AUTO_FENCE)
```

## YAKL behavior definitions

| Definition | Default | Effect |
| --- | --- | --- |
| `YAKL_PROFILE` | off | Enables explicit timer collection and timer queries. Timer start/stop fences the Kokkos runtime. |
| `YAKL_AUTO_PROFILE` | off | Enables YAKL's internal operation timers and automatically defines `YAKL_PROFILE`. |
| `YAKL_AUTO_FENCE` | off | Adds a `Kokkos::fence()` after YAKL launches and selected asynchronous operations. |
| `HAVE_MPI` | off | Includes MPI integration used by rank-aware output/autotuning and MPI-enabled build paths. |

The headers expose corresponding `inline constexpr bool` values: `yakl::yakl_profile`, `yakl::yakl_auto_profile`,
`yakl::yakl_auto_fence`, and `yakl::have_mpi`.

`yakl::yakl_mainproc()` returns true on every process without MPI support and only on rank zero of its communicator with
MPI support; its MPI overload accepts a communicator and defaults to `MPI_COMM_WORLD`. `yakl::my_basename(path)` returns the
final slash- or backslash-separated path component and is used to form automatic labels.

### `YAKL_PROFILE`

When enabled, `timer_start`, `timer_stop`, duration queries, count queries, and `timer_print` operate on YAKL's global timer.
When disabled, starts/stops/printing are no-ops, numeric duration queries return zero, and count returns zero. Enabling it can
substantially change timing and concurrency because each start and stop fences outstanding Kokkos work.

### `YAKL_AUTO_PROFILE`

This enables `YAKL_PROFILE` and instruments operations that contain automatic timer hooks, including many copies, launches,
reductions, and array transformations. It is intended for diagnosis and coarse profiling rather than low-overhead production
measurement. Explicit user timers remain available.

### `YAKL_AUTO_FENCE`

YAKL normally preserves Kokkos asynchronous execution. This definition fences after many YAKL kernels, fills, copies, and
componentwise operations, which can make asynchronous failures easier to localize. It can seriously reduce performance and
should not be needed for correctness in code with valid dependency and lifetime management. Operations that must return
host-visible results, timer boundaries, and `yakl::finalize()` synchronize regardless of this definition.

### `HAVE_MPI`

Define this only when MPI headers and libraries are available and the build is configured for MPI. It makes MPI declarations
visible to YAKL and allows rank-zero-only reporting paths to query `MPI_COMM_WORLD`. CMake users should set
`YAKL_HAVE_MPI=ON`, which finds and links MPI and supplies `HAVE_MPI` transitively through the `yakl` target. The PNetCDF
extension independently requires PNetCDF; merely defining `HAVE_MPI` does not provide that dependency.

## Kokkos definitions observed by YAKL

| Definition | YAKL effect |
| --- | --- |
| `KOKKOS_ENABLE_DEBUG` | Sets `yakl::kokkos_debug = true` and enables general validity, initialization, size, and overflow checks guarded by that constant. |
| `KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK` | Sets `yakl::kokkos_bounds_debug = true`; YAKL also ensures `KOKKOS_ENABLE_DEBUG` is defined. Enables index and range checks. |
| `KOKKOS_ENABLE_CUDA` | Selects CUDA-specific safe lambda-capture handling and CUDA event timing for autotuning. |
| `KOKKOS_ENABLE_HIP` | Selects HIP event timing for autotuning. |

Kokkos backend, architecture, compiler, and tuning definitions retain their normal Kokkos meanings. YAKL uses
`Kokkos::DefaultExecutionSpace` and its memory space for launches and the backing implementation of `yakl::DeviceSpace`.

### Debug-check contract

Debug macros control whether many checks execute, not whether the preconditions exist. Release builds may not diagnose:

- negative or unrepresentable extents and indices;
- invalid bounds, strides, or tile sizes;
- mismatched componentwise shapes;
- unallocated arrays passed to algorithms;
- dimension-product overflow;
- out-of-range indexing and linear-index unpacking.

Some safety checks are intentionally unconditional. In particular, allocation/free outside the YAKL lifecycle, live device
allocations at finalization, invalid file I/O state, and operations whose result cannot be meaningful may abort in every
build.

## Definitions supplied by YAKL

`YAKL_SCOPE(name,value)` creates a reference suitable for capture in a `KOKKOS_LAMBDA`; CUDA builds route it through a
no-inline helper to avoid problematic capture transformations. `YAKL_AUTO_LABEL()` produces a source-derived kernel label.
These macros are implementation conveniences available to users, but normal code can usually capture an array directly and
provide an explicit descriptive label.

## CMake and test-only switches

These are CMake variables rather than C/C++ preprocessor definitions:

| CMake option | Purpose |
| --- | --- |
| `YAKL_ENABLE_COVERAGE` | Build YAKL and consumers with GNU gcov instrumentation. |
| `YAKL_HAVE_MPI` | Find and link MPI and publicly define `HAVE_MPI` on the `yakl` target. |
| `YAKL_INDEX_BITS` | Logical index, loop-bound, and flattened-iteration width: `32` or `64` (default). |
| `YAKL_TEST_NETCDF` | Opt the unit suite into NetCDF tests when dependencies are available. |
| `YAKL_TEST_PNETCDF` | Opt the unit suite into PNetCDF tests when dependencies are available. |
| `YAKL_UNIT_LARGE_MEMORY` | Build tests requiring more than 4 GiB of device memory; forced off for 32-bit indices. |

Machine environment files may translate environment variables into these CMake options, but environment variables are not
automatically preprocessor definitions. The runtime pool environment variables are documented in
[getting started](getting-started.md#initconfig).

Configure with `-DYAKL_INDEX_BITS=32` when every logical array extent, flattened iteration count, loop bound, and index fits
in 32 bits. YAKL then exposes `yakl::index_t` as `std::int32_t` and `yakl::uindex_t` as `std::uint32_t`; the default 64-bit
configuration exposes the corresponding 64-bit types. Allocation sizes and byte offsets remain `size_t`. The setting is a
public compile definition on the `yakl` target, so linked C++ consumers receive the same choice and must be rebuilt when it
changes.

[API home](README.md) · [Getting started](getting-started.md) · [Memory](memory.md)
