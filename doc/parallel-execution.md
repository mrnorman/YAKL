# Parallel execution

[API home](README.md) · [Arrays](arrays.md) · [Algorithms](algorithms.md)

YAKL maps multidimensional loops to Kokkos range policies. Unsuffixed launchers use C-style indices and `_F` launchers use
Fortran-style indices. Ranks one through eight are supported.

## Loop specifications

`LoopSpec` is the C-style loop descriptor and `LoopSpec_F` is the Fortran-style loop descriptor. Neither has a style
template parameter. Both describe an inclusive lower bound, inclusive upper bound, and positive stride.

```cpp
yakl::LoopSpec a(10);       // 0:9
yakl::LoopSpec b(-3,7);     // -3:7
yakl::LoopSpec c(0,8,2);    // 0,2,4,6,8

yakl::LoopSpec_F fa(10);      // 1:10
yakl::LoopSpec_F fb(-3,7,2);  // -3,-1,1,3,5,7
```

The one-argument constructor is an extent. A `LoopSpec` used in general `Bounds` must describe at least one iteration; use
`SimpleBounds` when a zero extent and therefore an empty launch is required. Explicit `(lower,upper[,stride])` intervals must
be nonempty. Only positive strides are supported. `index_range()` returns `(upper-lower)/stride + 1`, so an exactly reachable
final upper bound is included. Bounds must be representable by `ptrdiff_t`.

The public data members `l`, `u`, and `s` hold the normalized values. `valid()` reports whether the stride is positive;
`index_range()` returns the trip count and requires a valid nonempty interval.

## Bounds

### General bounds

`Bounds<N>` accepts exactly `N` `LoopSpec` values. `Bounds_F<N>` accepts exactly `N` `LoopSpec_F` values. The general form invokes
the kernel with `ptrdiff_t` indices, preserving negative and arbitrary lower bounds.

```cpp
yakl::Bounds<3> bounds({-1,1},{0,nj-1},{2,ni-1,2});
yakl::parallel_for("general",bounds,KOKKOS_LAMBDA (ptrdiff_t k, ptrdiff_t j, ptrdiff_t i) {
  // k=-1..1, j=0..nj-1, i=2,4,... <= ni-1
});

yakl::Bounds_F<2> fbounds({-2,2},{3,9,3});
yakl::parallel_for_F("fortran",fbounds,KOKKOS_LAMBDA (ptrdiff_t j, ptrdiff_t i) {
  // j=-2..2, i=3,6,9
});
```

### Simple bounds

`SimpleBounds<N>` and `SimpleBounds_F<N>` accept extents rather than `LoopSpec` values. C-style indices run from zero;
Fortran-style indices run from one. Simple launchers pass `size_t` indices. A zero extent produces an empty launch.

```cpp
yakl::SimpleBounds<2> cb(ny,nx);    // [0,ny) x [0,nx)
yakl::SimpleBounds_F<2> fb(ny,nx);  // [1,ny] x [1,nx]
```

The bounds classes expose `nIter`, the total iteration count, and `unpack(linear,...)`, which converts the flattened C-order
iteration offset to style-correct indices. Dimension products use `size_t`; invalid negative extents and arithmetic overflow
are rejected when Kokkos debug checking is enabled.

`offs` contains internal row-major flattening offsets. General bounds also expose `lbounds` and `strides`. These members are
useful for device-side inspection, but callers should normally use `unpack` rather than reproduce its arithmetic.

## `parallel_for`

The primary overload family is:

```cpp
yakl::parallel_for([label,] bounds_or_extent, functor [, config]);
yakl::parallel_for_F([label,] bounds_or_extent, functor [, config]);
```

- `bounds_or_extent` may be a matching bounds object or one integral one-dimensional extent.
- The optional label is used by Kokkos tools, profiling, and autotuning. Without it, YAKL derives a source-location label.
- The functor must be callable with one index per dimension and must be valid for the Kokkos default execution space.
- The `_F` form only accepts Fortran-style bounds and produces one-based simple indices.
- Empty bounds return without launching a Kokkos kernel.
- Launch is asynchronous unless automatic fencing/profiling forces synchronization.

Always spell the qualified name `yakl::parallel_for`. On CUDA builds, an unqualified call made visible by `using` can be
misresolved as `Kokkos::parallel_for` through argument-dependent lookup when an argument is a Kokkos-associated type such as
`size_t` returned by a View.

## Launch configuration

`Config<MaxThreadsPerBlock>` combines a compile-time Kokkos launch bound with independent runtime tiles for each dimension:

```cpp
yakl::Config<> default_config;       // no thread limit and no tiling
yakl::Config<256> tiled(2,4);        // at most 256 threads/block, with a 2 x 4 tile

yakl::parallel_for("tiled",bounds,KOKKOS_LAMBDA (size_t j, size_t i) {
  // body
},tiled);
```

Tile one in every dimension takes the ordinary untiled path and introduces no tiling loop. Any dimension greater than one
creates a multidimensional Cartesian tiling. Each Kokkos policy iteration processes one tile and runs all valid points in
that tile serially within the policy work item. Edge tiles are shortened and every logical point runs exactly once. Runtime
tiles must be positive. Up to eight tile dimensions may be supplied; omitted trailing dimensions default to one.

`MaxThreadsPerBlock == 0` leaves the backend unconstrained. A nonzero value is passed as a Kokkos `LaunchBounds` maximum; it
does not itself choose a team size. `Config::Thr` exposes the compile-time value and `config.tiles[d]` exposes each runtime
tile.

## Autotuning

`yakl::autotune::parallel_for` and `parallel_for_F` require an explicit stable string label. For each label and shape,
autotuning considers the untiled Kokkos-default configuration followed by launch bounds `64`, `128`, `256`, `512`, and
`1024`. Each configuration is visited five times; the first timing is discarded and later timings are accumulated. Once
all configurations have been visited, subsequent calls use the best measured configuration.

```cpp
yakl::autotune::parallel_for("stencil",bounds,KOKKOS_LAMBDA (size_t j, size_t i) {
  out(j,i) = in(j,i-1) + in(j,i) + in(j,i+1);
});
```

The identity includes the label and bound dimensions. Use the same label for the same kernel body and logical workload;
do not deliberately combine unrelated kernels. Timing uses CUDA or HIP events on those backends and fenced wall-clock time
elsewhere. `yakl::autotune::print_best()` prints the selected `Config<threads>` and its speedup relative to the leading
Kokkos-default configuration; `yakl::finalize()` calls it.
If a label has not completed its full tuning cycle, the report marks it as incomplete and uses the best completed timed
sample available. A label that has only reached its discarded warmup reports that no timed sample exists. Finalization
does not force the remaining configurations to run and does not treat partial tuning as an error.
After printing the report, `yakl::finalize()` clears all autotuning contexts. A later `yakl::init()` therefore begins a new
tuning interval rather than reusing complete or partial measurements from the previous lifecycle interval.

Autotuning is a host-thread-confined facility. Its process-local state is shared by every autotuned label and has no
internal synchronization. Do not call any `yakl::autotune::parallel_for`, `parallel_for_F`, or `print_best` operation
concurrently from multiple application host threads, even when labels differ. Complete or join all such callers before
`yakl::finalize()`. Ordinary non-autotuned YAKL launchers are not subject to this autotune-state restriction.

## Host/device and lifetime rules

The launcher copies bounds and the functor into Kokkos execution. Capture only device-copyable values. Device arrays and
small inline arrays are safe captures; host containers, `std::string`, and host pointers are not. The owning allocation for
every captured array or device-created unmanaged alias must remain alive until the kernel completes. A later operation in
the same ordered execution space may provide dependency ordering, but destruction/finalization requires an explicit fence.

See [compile-time configuration](configuration.md) for `YAKL_AUTO_FENCE`, profiling, and debug behavior.

[Previous: Arrays](arrays.md) · [API home](README.md) · Next: [Algorithms](algorithms.md)
