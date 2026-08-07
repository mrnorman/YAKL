# Random numbers, live-out scalars, and timers

[API home](README.md) · [Algorithms](algorithms.md) · [Configuration](configuration.md)

## `Random`

`yakl::Random` is an allocation-free Philox4x32-10 pseudo-random generator usable in host and device code. It is designed
for simulation, not cryptography.

```cpp
yakl::parallel_for("random",n,KOKKOS_LAMBDA (size_t i) {
  yakl::Random rng(12345,static_cast<uint64_t>(i));
  uniform(i) = rng.gen_uniform<double>();
  normal (i) = rng.gen_normal<double>();
});
```

The 64-bit seed identifies an experiment. The 64-bit stream ID identifies an independent logical consumer inside that
experiment, commonly a stable global cell, particle, ensemble-member, or chain ID. A `(seed,stream_id)` pair deterministically
restarts the same sequence on host and device. Do not seed from thread scheduling identifiers if reproducibility across launch
configurations is required. Each `Random` object contains mutable state and must be private to one logical consumer; sharing
one object between threads is a data race.

### Generator API

| Member | Distribution and semantics |
| --- | --- |
| `set_seed(seed,stream_id)` | Restart at the beginning of the selected deterministic stream and discard a cached normal. |
| `gen_uniform<uint64_t>()` | One raw 64-bit output. |
| `gen_uniform<T>()` | Floating-point uniform value in `[0,1)`; `T` defaults to `float`. |
| `gen_uniform<T>(lb,ub)` | Floating-point uniform value in `[lb,ub)`; equal bounds return that bound. |
| `gen_normal<T,CacheSpare>(mean,stddev)` | Normal distribution; defaults are `T=float`, `CacheSpare=true`, mean zero, standard deviation one. |
| `gen_bernoulli(probability)` | Boolean result with the requested probability; default `0.5`. |
| `gen_exponential<T>(rate)` | Exponential distribution with positive rate; default `T=float`, rate one. |
| `gen_lognormal<T,CacheSpare>(normal_mean,normal_stddev)` | Exponential of a normal draw using underlying-normal parameters. |

Floating uniform generation uses the scalar type's mantissa width and never returns the upper endpoint. Ranged generation
uses overflow-safe interpolation, including for finite intervals such as `[-max,+max)`. `CacheSpare=true`
uses both Box-Muller outputs across calls. Setting it false discards any old spare and the newly produced second value, giving
a fixed two-uniform consumption per normal call at higher cost. Zero standard deviation returns the mean without consuming
random state. Non-finite parameters, reversed bounds, out-of-range probabilities, negative standard deviations, and
nonpositive rates are rejected when debug checks are enabled.

Copying a generator copies its exact sequence position and cached normal state. The original and copy will then produce
identical future values until called differently; copying is not a way to create a new stream.

## `ScalarLiveOut`

`ScalarLiveOut<T>` owns one `T` in `yakl::DeviceSpace` and is convenient when a device lambda must update a scalar that host
code later reads.

```cpp
yakl::ScalarLiveOut<int> result(0);
yakl::parallel_for("find",n,KOKKOS_LAMBDA (size_t i) {
  if (condition(i)) result() = static_cast<int>(i);
});
int host_result = result.hostRead();
```

`operator()`, `get`, and scalar assignment access the device-side value and are host/device callable, but ordinary host code
must not dereference them when device memory is inaccessible. Use `hostRead()` for a synchronized value and `hostWrite(v)`
to submit a one-element device write. Copies share the underlying allocation. Construction/destruction follows the same
`yakl::init()`/`yakl::finalize()` lifetime as other `DeviceSpace` arrays. Construct the owning object in host code before
capturing it in a kernel; device-side construction cannot allocate its storage. Concurrent writes need atomics or another
valid race-free algorithm.

## Timers

The recommended timer API is label-based:

```cpp
yakl::timer_start("time step");
// launches or host work
yakl::timer_stop("time step");

double last = yakl::timer_get_last_duration("time step");
size_t hits = yakl::timer_get_count("time step");
yakl::timer_print();
```

Available queries are:

- `timer_get_last_duration(label)`;
- `timer_get_accumulated_duration(label)`;
- `timer_get_min_duration(label)`;
- `timer_get_max_duration(label)`; and
- `timer_get_count(label)`.

Timers are active only with `YAKL_PROFILE` (or `YAKL_AUTO_PROFILE`). Disabled timer starts, stops, and printing do nothing;
queries return zero. Enabled start/stop calls fence Kokkos before reading the clock. Timing therefore measures completed work
but introduces synchronization. Elapsed time is measured with the monotonic `std::chrono::steady_clock`.

Timers must be perfectly nested. The most recently started label must be the next one stopped. Stopping an empty stack,
stopping a different label, or querying an unknown label is an error. Labels are compared by string, so distinct labels are
not merged merely because their hashes collide. Reusing one label accumulates hits, last/min/max duration, and total duration.
Recursive use of the same label records every invocation with its own start time and inclusive duration. Nested labels are
displayed as parent/child relationships; a label observed under multiple parents is reported accordingly.

Timer operations are thread-safe. Nesting is tracked independently for each host thread, while records with the same label
are aggregated across threads. Duration and count queries and printed reports observe a consistent snapshot of completed
timer updates. Direct unsynchronized access to `Toney`'s public storage is not part of this guarantee.

`timer_print()` prints the current main timer hierarchy. `yakl::finalize()` prints it automatically when profiling is active.
Finalization then clears timer records and per-thread nesting stacks so a later YAKL lifecycle interval starts with no timer
history from the preceding interval.
`yakl::Toney` is the underlying public timer type and exposes `start`, `stop`, duration/count queries, `print_main`, and
`print`; direct use is possible, but the global wrappers supply the required device fence and compile-time disabling.

[Previous: Algorithms](algorithms.md) · [API home](README.md) · Next: [Memory management](memory.md)
