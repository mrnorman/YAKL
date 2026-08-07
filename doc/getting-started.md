# Getting started and lifecycle

[API home](README.md) · [Arrays](arrays.md) · [Parallel execution](parallel-execution.md)

## Requirements and integration

YAKL requires C++20 and supports Kokkos 4.7.00 through the current supported release. Add Kokkos and YAKL to the build and
link the `yakl` target:

```cmake
add_subdirectory(${KOKKOS_HOME} ${KOKKOS_BIN})
add_subdirectory(${YAKL_HOME} ${YAKL_BIN})

add_executable(example example.cpp)
target_link_libraries(example PRIVATE yakl Kokkos::kokkos)
```

Applications normally include `YAKL.h`. The extension headers `extensions/YAKL_netcdf.h` and
`extensions/YAKL_pnetcdf.h` require their corresponding libraries; see [I/O](io.md).

## Lifecycle

Kokkos owns the execution runtime. The application must initialize Kokkos first, initialize YAKL next, destroy all
YAKL-device allocations before finalizing YAKL, and finalize Kokkos last.

```cpp
int main(int argc, char **argv) {
  Kokkos::initialize(argc,argv);
  {
    yakl::init();
    {
      yakl::Array<double *> a("a",1024);
      yakl::parallel_for("initialize",a.size(),KOKKOS_LAMBDA (size_t i) {
        a(i) = static_cast<double>(i);
      });
    } // a must be destroyed before yakl::finalize()
    yakl::finalize();
  }
  Kokkos::finalize();
}
```

`yakl::init()` and `yakl::finalize()` are single-caller lifecycle operations. Call them from the application's controlling
host thread, outside every application-level host threaded region. `init()` must finish before entering an OpenMP region,
starting `std::thread` workers that use YAKL, or launching other host work that can call YAKL. Before `finalize()`, leave those
regions and join or otherwise quiesce every application host thread that can execute a YAKL operation. Kokkos may keep its
own internal worker threads alive; the requirement is that no application work is concurrently using YAKL.

This is an unconditional API precondition, not a debug-only check. Never call `init()` or `finalize()` concurrently, and do
not overlap either call with array construction/destruction, raw allocation/deallocation, launches, timer calls, or other
YAKL operations. The fence inside `finalize()` waits for submitted Kokkos execution but does not join application host
threads or make a concurrent lifecycle transition safe.

`yakl::init()` and allocation through `yakl::DeviceSpace` require an initialized Kokkos runtime. Allocation or
deallocation through `DeviceSpace` outside the interval from `yakl::init()` through `yakl::finalize()` is a fatal error.
`yakl::finalize()` fences, rejects live `DeviceSpace` allocations, prints enabled timers and autotune results, and releases
the pool. A second `init` or `finalize` is ignored with a warning in debug builds, but should not be used as normal control
flow.

## `InitConfig`

`InitConfig` uses chainable setters. Values are copied into the returned configuration object.

```cpp
yakl::InitConfig config;
config = config.set_pool_enabled(true)
               .set_pool_size_mb(8192)
               .set_pool_block_bytes(4096);
yakl::init(config);
```

| Member | Meaning |
| --- | --- |
| `set_pool_enabled(bool)` | Explicitly enable or disable the device pool. `false` is honored even with the default size. |
| `set_pool_size_mb(size_t)` | Set the pool capacity in MiB. Zero means use the default/environment configuration. |
| `set_pool_block_bytes(size_t)` | Explicitly set allocation granularity; it must be a positive multiple of Kokkos memory alignment. |
| `get_pool_setting()` | Return `PoolSetting::Default`, `Enabled`, or `Disabled`. |
| `get_pool_enabled()` | True only when the setting was explicitly enabled. |
| `get_pool_size_mb()` | Return the configured size, where zero still means unspecified. |
| `get_pool_block_bytes()` | Return the configured block size. |

When no explicit nonzero pool size is supplied, these environment variables are read:

| Variable | Behavior |
| --- | --- |
| `GATOR_DISABLE` | `yes`, `YES`, `1`, `true`, `TRUE`, or `T` disables the pool. |
| `GATOR_INITIAL_MB` | Positive pool size in MiB; the default is 4096 MiB. |
| `GATOR_BLOCK_BYTES` | Positive block size divisible by Kokkos memory alignment; the default is 4096 bytes. |

An explicit `set_pool_enabled` selection overrides `GATOR_DISABLE`. An explicit `set_pool_block_bytes` value always
overrides `GATOR_BLOCK_BYTES`, even when the pool size comes from the environment or the 4096 MiB default. An explicit
nonzero pool size overrides `GATOR_INITIAL_MB`.

## First multidimensional kernel

```cpp
using yakl::Array;
using yakl::Bounds;
using yakl::SimpleBounds;

Array<double **> a("a",128,256);
yakl::parallel_for("fill",SimpleBounds<2>(128,256),KOKKOS_LAMBDA (size_t j, size_t i) {
  a(j,i) = 10.0*j + i;
});

yakl::parallel_for("interior",Bounds<2>({1,126},{2,253}),KOKKOS_LAMBDA (ptrdiff_t j, ptrdiff_t i) {
  a(j,i) += 1.0;
});
```

Always qualify `yakl::parallel_for`. CUDA compiler argument-dependent lookup can ambiguously resolve an unqualified call,
particularly when a `size_t` bound originates from a Kokkos View.

## Compile-time behavior controls

See [Compile-time configuration](configuration.md) for the complete reference and build guidance.

| Definition | Effect |
| --- | --- |
| `KOKKOS_ENABLE_DEBUG` | Enables general precondition and overflow checks exposed as `yakl::kokkos_debug`. |
| `KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK` | Enables index/bounds checks exposed as `yakl::kokkos_bounds_debug`; also enables Kokkos debug. |
| `YAKL_AUTO_FENCE` | Fences after YAKL launches and selected asynchronous operations. Useful for debugging, costly for production. |
| `YAKL_PROFILE` | Enables the explicit `timer_*` API. |
| `YAKL_AUTO_PROFILE` | Times selected YAKL operations automatically and implies `YAKL_PROFILE`. |
| `HAVE_MPI` | Enables MPI-aware rank handling where compiled into YAKL. |

Checked preconditions are still part of the API contract when checking is disabled. Errors generally use `Kokkos::abort`,
so callers should not expect to catch them as C++ exceptions.

## Synchronization rules

- `parallel_for`, scalar array assignment, type conversion, and device-side copies only fence automatically with
  `YAKL_AUTO_FENCE`.
- `createHostCopy()` always fences before returning host-readable data.
- `deep_copy_to(host_array)` fences because the destination is `Kokkos::HostSpace`.
- reductions that return a host scalar synchronize through Kokkos reduction semantics.
- timers fence at start and stop when profiling is enabled so measured intervals represent completed work.
- After application host threads are quiescent, `yakl::finalize()` fences submitted Kokkos work before checking allocation
  lifetime and freeing the pool.

[API home](README.md) · Next: [Arrays and indexing](arrays.md)
