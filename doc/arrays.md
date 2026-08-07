# Arrays and indexing

[API home](README.md) · [Getting started](getting-started.md) · [Algorithms](algorithms.md)

## Choosing an array type

| Type | Shape | Index convention | Layout | Storage |
| --- | --- | --- | --- | --- |
| `Array<T, MemSpace>` | Runtime | zero based | `Kokkos::LayoutRight` | Kokkos allocation |
| `Array_F<T, MemSpace>` | Runtime | arbitrary signed lower bounds | `Kokkos::LayoutLeft` | Kokkos allocation |
| `SArray<T, dims...>` | Compile time | zero based | last index fastest | inline in the object |
| `SArray_F<T, Bnds...>` | Compile time | arbitrary signed lower bounds | first index fastest | inline in the object |

Use dynamic arrays for application data and `SArray` variants for small fixed-size values, such as matrices or per-thread
temporaries. `Array` defaults to `yakl::DeviceSpace`; explicitly request `Kokkos::HostSpace` for ordinary host access.

## Supporting types

Dynamic-array rank is encoded in a Kokkos-style data type:

```cpp
yakl::Array<float ***> a("a",nz,ny,nx);
using Rank4Double = typename yakl::ViewType<double,4>::type; // double ****
yakl::Array<Rank4Double> b("b",nt,nz,ny,nx);
```

`ViewType<Value, Rank>::type` adds `Rank` pointer levels. Const qualification on the value type creates a read-only view.
`yakl::COLON` is an alias of `Kokkos::ALL` and denotes a complete retained dimension in slicing.

For compile-time Fortran bounds, `yakl::Bnds{lower,upper}` is an inclusive structural template argument:

```cpp
yakl::SArray_F<double,yakl::Bnds{-2,2},yakl::Bnds{1,3}> stencil;
```

## Dynamic arrays

### `Array`

`template<class DataType, class MemSpace = yakl::DeviceSpace> class Array` publicly derives from
`Kokkos::View<DataType,Kokkos::LayoutRight,MemSpace>`. It inherits compatible Kokkos View constructors, assignment,
metadata, and `operator()`. A labeled constructor owns an allocation; a pointer constructor is unmanaged.

```cpp
yakl::Array<double **> device("state",ny,nx);
yakl::Array<double **,Kokkos::HostSpace> host("host_state",ny,nx);
yakl::Array<double **,Kokkos::HostSpace> alias(host.data(),ny,nx); // unmanaged
```

Indices are zero based and the last index varies fastest. Rank is `ArrayType::rank()` and the usual Kokkos View members,
including `extent`, `stride`, `size`, `span`, `data`, `label`, `is_allocated`, and `use_count`, remain available.

### `Array_F`

`Array_F` also derives from a Kokkos View, but uses `LayoutLeft` and stores a lower bound for every dimension. It does not
inherit arbitrary View constructors because those could lose lower-bound metadata. Construct each dimension with its
nested inclusive `AB` bounds type; one argument means a default lower bound of one.

```cpp
using F2 = yakl::Array_F<double **>;
F2 a("a",F2::AB{-3,4},F2::AB{1,8});
F2 b("b",{16},{-5,5});                  // [1:16,-5:5]
F2 alias(a.data(),{-3,4},{1,8});        // unmanaged pointer view
```

All integral index and bound types are accepted when representable by `ptrdiff_t`. `operator()` subtracts the stored lower
bounds before accessing the underlying View. The first index varies fastest. Copy and move operations preserve both the
Kokkos allocation relationship and lower bounds.

### Common dynamic-array members

The following members apply to both styles unless noted otherwise.

| Member | Result and semantics |
| --- | --- |
| `operator=(scalar)` | Fill every element with an arithmetic scalar using `Kokkos::deep_copy`. |
| `clone_object<Space,Value>()` | New owning allocation with the same shape, bounds, and label; data is not copied. |
| `deep_copy_to(dst)` | Copy all elements. Total sizes must match; shape and lower bounds need not match. |
| `createDeviceObject()` / `createHostObject()` | New uninitialized allocation with matching metadata. |
| `createDeviceCopy()` / `createHostCopy()` | Allocate and deep-copy into the named space. A host copy is ready to read. |
| `as<Scalar>()` | Allocate the same shape/space and componentwise-convert values to `Scalar`. |
| `extents()` / `lbounds()` / `ubounds()` | Return inline arrays of per-dimension metadata. For `_F`, the metadata array itself is indexed `1..rank`. |
| `begin()` / `end()` | Raw contiguous pointer range in the array's own memory space. It does not make device memory host-accessible. |
| `unpack_global_index(i)` | Convert a contiguous linear offset to valid style-correct indices. `_F` includes arbitrary lower bounds. |
| `get_View()` | Reference to the underlying Kokkos View base. |
| stream `operator<<` | Create a host copy and print all values; this is synchronous and intended for diagnostics. |

`as<Scalar>()` preserves `Array_F` lower bounds. The returned value of componentwise conversion is a distinct allocation.

### Shape transforms and aliases

These operations never copy elements:

| Member | Meaning |
| --- | --- |
| `slice<NewRank>(...)` | Remove complete slow dimensions and fix each removed dimension at one index. |
| `subset_slowest_dimension(l,u)` | Inclusive contiguous subset of the slowest dimension. For `Array`, the one-argument form `subset_slowest_dimension(n)` selects `[0,n-1]`; for `Array_F`, `subset_slowest_dimension(u)` selects `[current_lower,u]`. |
| `reshape(...)` | Reinterpret the same contiguous elements with a new shape of equal total size. |
| `collapse()` | One-dimensional zero-based alias for `Array`, or one-based alias for `Array_F`. |
| `flatten()` | Alias for `collapse()`; `Array_F::flatten(lb)` chooses an arbitrary lower bound. |

For `Array`, the slowest dimension is dimension zero. For `Array_F`, it is the last dimension. Consequently, `Array::slice`
removes leading dimensions while `Array_F::slice` removes trailing dimensions. Every original dimension still has a slice
argument. Arguments for retained dimensions are accepted for backward compatibility but ignored and treated as
`Kokkos::ALL`; only removed dimensions' integral indices select data. YAKL does not provide partial-range slicing through
this member API.

```cpp
yakl::Array<double ***> c("c",nz,ny,nx);
auto plane = c.slice<2>(k,yakl::COLON,yakl::COLON); // shape [ny,nx]
auto lower = c.subset_slowest_dimension(0,nz/2-1);
auto flat  = c.flatten();

using F3 = yakl::Array_F<double ***>;
F3 f("f",{-2,2},{1,ny},{10,19});
auto fplane = f.slice<2>(yakl::COLON,yakl::COLON,12); // retains first two dimensions and their lower bounds
```

#### Ownership of transformed views

When a transform is called in host code, YAKL retains the source Kokkos allocation record in the result. The result can
therefore outlive the particular source object, and `use_count()` reflects that retained ownership. When the same
device-callable method executes in device code, the result is unmanaged: device code must not touch host-only allocation
tracking, and the original owning allocation must remain alive until all work using the alias completes. Copies of an
already-unmanaged view remain unmanaged.

All transforms require contiguous layout assumptions supplied by `Array`/`Array_F`; `reshape` additionally requires exactly
the same element count.

## Static arrays

`SArray<T,dims...>` and `SArray_F<T,Bnds...>` store all elements directly inside the object. Their constructors and methods
are host/device callable and they require no YAKL initialization or allocation. Dimensions must be nonempty valid
compile-time ranges.

```cpp
yakl::SArray<double,3,3> m;
yakl::SArray_F<int,yakl::Bnds{-1,1},yakl::Bnds{2,4}> fm;

KOKKOS_LAMBDA (size_t i) {
  yakl::SArray<double,3> local;
  local = 0.0;
  local(1) = static_cast<double>(i);
};
```

Both expose `value_type`, `const_value_type`, `non_const_value_type`, `rank`, `size`, `data`, `begin`, `end`, `extent`,
`extents`, `lbounds`, `ubounds`, `unpack_global_index`, `span_is_contiguous`, and `is_allocated`. `TypeAs<New>` preserves
the shape while changing the scalar type. Scalar assignment fills all elements. Stream output is host-only in practice.

Unlike dynamic arrays, assigning or passing an `SArray` by value copies its elements. Keep fixed arrays small enough for the
target execution space's stack/register constraints.

## Generic array code

Traits allow one implementation to describe a family of ranks and styles:

```cpp
template <class A>
  requires yakl::is_Array<A> && A::is_cstyle && (A::rank() == 2)
void launch(A const &a);

template <class A>
  requires yakl::is_SArray<A> && A::is_fstyle && (A::rank == 2)
KOKKOS_INLINE_FUNCTION auto trace(A const &a) {
  typename A::non_const_value_type result = 0;
  for (ptrdiff_t i = a.lbounds()(1); i <= a.ubounds()(1); i++) result += a(i,i);
  return result;
}
```

Dynamic rank is the Kokkos static function `rank()`; static-array rank is a static data member `rank`.

## Preconditions and failure behavior

- Indexing must respect each object's actual lower and upper bounds.
- Dynamic dimensions, products, and pointer constructors are validated only when the corresponding debug mode is enabled.
- `Array_F` dimensions are inclusive and require `upper >= lower`; negative lower bounds are valid.
- `ubounds()` on an empty C-style dimension is undefined and rejected in debug mode.
- Raw pointer constructors do not own memory. The pointer must remain valid, correctly aligned, in the declared memory space,
  and large enough for the full shape.
- No array API implicitly migrates data between host and device; use the explicit copy methods.

[Previous: Getting started](getting-started.md) · [API home](README.md) · Next: [Parallel execution](parallel-execution.md)
