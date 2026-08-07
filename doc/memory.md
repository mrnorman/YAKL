# Memory management

[API home](README.md) · [Getting started](getting-started.md) · [Arrays](arrays.md)

Most users should allocate `Array` objects and let their Kokkos View ownership manage deallocation. This page documents the
lower-level interfaces and their lifecycle constraints.

## `DeviceSpace`

`yakl::DeviceSpace` satisfies the Kokkos memory-space interface and uses
`Kokkos::DefaultExecutionSpace::memory_space` as its underlying device storage. It routes allocation through YAKL's optional
pool while remaining usable as a Kokkos View memory-space template argument:

```cpp
Kokkos::View<double *,yakl::DeviceSpace> view("view",n);
yakl::Array<double *,yakl::DeviceSpace> array("array",n);
```

Kokkos deep copies are supported between `DeviceSpace`, `Kokkos::HostSpace`, and the default execution space's memory space.
On accelerator builds, host code must not directly dereference `DeviceSpace` pointers. On host-only builds, Kokkos's memory
access rules may make the spaces mutually accessible.

Every nonzero allocation through `DeviceSpace` increments YAKL's live-allocation count. Allocation and deallocation outside
the active YAKL lifecycle abort. `yakl::finalize()` fences, then aborts if any such allocation remains; this structurally
prevents pool-backed pointers from surviving release of the pool.

## Pool behavior

The device pool is enabled by default and has a default capacity of 4 GiB. It obtains one backing allocation from Kokkos and
serves variable-size requests rounded up to a configured block size. Free blocks can be reused. The block size must be a
positive multiple of `Kokkos::Impl::MEMORY_ALIGNMENT`, making each returned block boundary Kokkos-aligned.

The pool does not grow. If a request cannot fit in a single free gap, allocation aborts even if total free bytes across
separate gaps would be sufficient. Disable the pool to route each request directly through `Kokkos::kokkos_malloc/free`, or
increase its size; see [`InitConfig` and environment variables](getting-started.md#initconfig).

Pool bookkeeping is protected by a recursive mutex, so concurrent host allocation/free calls do not corrupt metadata. The
allocator does not fence device execution on each allocation or free. Users must preserve normal Kokkos lifetime ordering;
the finalization fence only protects the final lifecycle boundary.

## `alloc_device` and `free_device`

```cpp
void *ptr = yakl::alloc_device(bytes,"field storage");
// ...
yakl::free_device(ptr,"field storage");
```

These are advanced raw allocation functions used by `DeviceSpace`. The exact base pointer returned by `alloc_device` must be
passed once to `free_device`, during the same initialized YAKL interval and under the same pool setting. Zero-byte pool
allocation may return `nullptr`, but `free_device(nullptr,...)` is not a valid deallocation. Prefer Kokkos Views or YAKL
arrays so ownership is automatic.

## `LinearAllocator`

`LinearAllocator` is the host-side implementation behind the pool and may also be constructed with user callbacks:

```cpp
yakl::LinearAllocator pool(bytes,block_bytes,allocate_callback,free_callback,zero_callback,"name",oom_message);
```

It owns one contiguous backing allocation and uses first-fit placement over address-ordered live allocation records.
Requests are rounded to whole blocks. Its notable public API is:

| Member | Meaning |
| --- | --- |
| `allocate(bytes,label)` | Return an aligned block or abort when no gap fits; zero bytes returns `nullptr`. |
| `free(ptr,label)` | Release an exact allocation base and return its rounded byte count. |
| `iGotRoom(bytes)` | Report whether a request currently fits in a gap. |
| `thisIsMyPointer(ptr)` | Report whether an address lies in the usable pool range; this does not prove it is an allocation base. |
| `initialized()` | Whether backing storage exists. |
| `poolSize()` | Usable rounded pool capacity in bytes. |
| `numAllocs()` | Number of live suballocations. |
| `getPtr(blockIndex)` | Address of a block; intended for allocator/debug use. |
| `printAllocsLeft()` | Print live labels, rounded sizes, offsets, and pointers. |
| `finalize()` | Release backing storage and reset the object. |

The class is movable, not copyable. Pool size and block size must be positive, all callbacks must be nonempty, and arithmetic
must fit in `size_t`. `requiredAlignment` exposes the Kokkos alignment used by validation. Destroying/finalizing a pool with
live suballocations can only emit a debug warning; direct users are responsible for ensuring none are used afterward.

## Fortran allocation API

The `gator_mod` Fortran module provides generic `gator_allocate` and `gator_deallocate` interfaces for ranks one through
seven and these scalar families:

- default integer and 8-byte integer;
- default real and 8-byte real;
- default complex and 8-byte complex; and
- default logical.

```fortran
use gator_mod
real(8), pointer :: field(:,:)
integer :: dims(2), lbounds(2)

call gator_init()
dims    = [128,256]
lbounds = [-2,1]
call gator_allocate(field,dims,lbounds)
! use field(-2:125,1:256) in a memory model where the allocation is accessible
call gator_deallocate(field)
call gator_finalize()
```

`dims` is required and every extent must be positive. Optional `lbounds_in` selects arbitrary lower bounds; otherwise
Fortran lower bounds default to one. Byte-count products are computed as `c_size_t` with overflow checks. Zero-element
allocations are deliberately rejected.

Allocation requires an unassociated pointer; it does not silently replace an existing target. Deallocation requires an
associated pointer whose base address is exactly one returned by `gator_allocate`; slices, interior pointers, foreign
allocations, null pointers, and double frees are rejected. The pointer is nullified after successful deallocation. All
Fortran allocations must be released before `gator_finalize`.

`gator_init` initializes Kokkos and/or YAKL only if the surrounding application has not already done so. `gator_finalize`
finalizes only the runtimes that `gator_init` itself acquired, so it is safe for a parent application to own Kokkos. Calls
must be paired; repeated initialization or unmatched finalization aborts.

The low-level C bindings are `gatorInit`, `gatorFinalize`, `gatorAllocate(size_t)`, and `gatorDeallocate(void*)`. They use a
registry to enforce base-pointer ownership and are intended primarily for the generated Fortran generic wrappers.

## Lifetime checklist

1. Initialize Kokkos, then YAKL.
2. Create all `DeviceSpace` owners only inside that interval.
3. Keep owners alive until every queued kernel using them or their unmanaged aliases is complete.
4. Destroy every owner and deallocate every Fortran/raw allocation.
5. Call `yakl::finalize()` while Kokkos is still initialized; its fence validates and closes the interval.
6. Finalize Kokkos.

[Previous: Random and timers](random-timers.md) · [API home](README.md) · Next: [NetCDF and PNetCDF](io.md)
