# Intrinsics and componentwise operations

[API home](README.md) · [Arrays](arrays.md) · [Parallel execution](parallel-execution.md)

YAKL provides two related algorithm namespaces:

- `yakl::intrinsics` supplies Fortran-like inquiries, reductions, selection, and small-matrix helpers.
- `yakl::componentwise` supplies elementwise C++ operators and math functions that return a new array.

Static-array overloads are `KOKKOS_INLINE_FUNCTION` and run directly in host or device code. Dynamic-array overloads are
host launch functions: they submit Kokkos kernels or reductions in the dynamic array's execution space.

## Intrinsics

Use explicit namespace qualification to avoid collisions with standard math functions:

```cpp
auto total = yakl::intrinsics::sum(a);
auto where = yakl::intrinsics::maxloc(a);
auto mask  = yakl::intrinsics::merge(positive,negative,condition);
```

### Element and selection operations

| Function | Inputs | Result |
| --- | --- | --- |
| `abs(a)` | `Array` or `SArray` family | Same array type and shape, with `std::abs` applied. |
| `sign(a,b)` | matching arrays or arithmetic scalars | Magnitude of `a` with a negative sign where `b < 0`; zero is nonnegative. |
| `merge(t,f,cond)` | matching arrays or arithmetic scalars | `t` where `cond` is true, otherwise `f`. |
| `same_shape(a,b)` | array-like values | True when ranks and every extent match. |

Dynamic `sign` inputs must have identical types and shape. Dynamic `merge` requires `t` and `f` to share a type and all
three arrays to have identical shapes. `merge` also requires every dynamic operand to have the same C/Fortran array style
and Kokkos memory space; mixing `Array` with `Array_F` or host with device storage is a compile-time error. Static shapes
and styles are constrained at compile time where possible. Returned dynamic arrays are new allocations shaped like the
first data operand.

### Inquiry functions

| Function | Meaning |
| --- | --- |
| `allocated(a)`, `associated(a)` | Forward to `a.is_allocated()`. Static arrays always report true. |
| `size(a)` | Total element count. |
| `size(a,i)` | Extent of dimension `i`; the dimension-number argument follows the metadata array's style. |
| `shape(a)` | Same result as `a.extents()`. |
| `lbound(a)`, `ubound(a)` | Inline arrays containing every lower or upper bound. |
| `lbound(a,i)`, `ubound(a,i)` | One lower or upper bound. |
| `epsilon(a)` | `numeric_limits<non_const_value_type>::epsilon()`. |
| `huge(a)` | `numeric_limits<non_const_value_type>::max()`. |
| `tiny(a)` | `numeric_limits<non_const_value_type>::min()`. |

For C-style arrays, bound/shape result arrays are zero based. For Fortran-style arrays, result arrays are indexed from one
through rank. `tiny` is the smallest positive normalized floating-point value, following `numeric_limits::min`, not the most
negative value.

### Reductions and locations

| Function | Result |
| --- | --- |
| `any(a)` | Logical OR of all components. |
| `all(a)` | Logical AND of all components. |
| `count(a)` | `size_t` count of components that convert to true. |
| `sum(a)` | Additive reduction in the non-const value type. |
| `product(a)` | Multiplicative reduction in the non-const value type. |
| `minval(a)`, `maxval(a)` | Smallest or largest value. |
| `minloc(a)`, `maxloc(a)` | Style-correct multidimensional indices of the first matching extremum in contiguous memory order. |

Dynamic reductions use `Kokkos::parallel_reduce`. The returned scalar or location is host-visible when the call returns.
Location results use `unpack_global_index`: `Array` returns zero-based indices; `Array_F` and `SArray_F` return their actual
possibly negative lower-bound-relative indices. If an extremum occurs multiple times, the lowest contiguous linear offset
wins. Min/max values and locations require nonempty inputs. Floating-point reduction association is backend-dependent, so
bitwise-identical sums/products across different execution configurations are not promised. With Kokkos debug enabled,
`minloc` and `maxloc` reject any input containing NaN. Release builds omit the NaN check entirely; callers must provide
NaN-free input when a meaningful location is required.

### Small-matrix operations

These functions operate only on C-style `SArray` values and are host/device callable:

| Function | Meaning |
| --- | --- |
| `matmul_rc(a,b)` | Conventional row/column matrix-vector or matrix-matrix multiplication. |
| `matmul_cr(a,b)` | Multiplication for inputs whose logical row/column interpretation is transposed in storage. |
| `transpose(a)` | Transpose a rank-two static array. |
| `matinv(a)` | Invert a square static matrix with partial pivoting. |

Matrix dimensions are compile-time checked. `matinv` accepts only floating-point element types and returns the same scalar
type and dimension. With Kokkos debug enabled, it rejects a pivot no larger than
`epsilon * matrix_scale * dimension`; without debug checking, singular input violates the precondition and may produce
infinities or NaNs.

## Componentwise API

Bring the namespace into a narrow scope when operator syntax is desired:

```cpp
using namespace yakl::componentwise;
auto normalized = abs(a) / maximum_scale; // each operation creates its own result
auto mask = (a >= lower) && (a <= upper);
```

### Binary operators

`+`, `-`, `*`, `/`, `<`, `>`, `<=`, `>=`, `==`, `!=`, `&&`, and `||` accept:

- two arithmetic scalars;
- one arithmetic scalar and one `SArray`/`SArray_F`, in either order;
- one arithmetic scalar and one `Array`/`Array_F`, in either order;
- two static arrays of matching rank, size, and shape; or
- two dynamic arrays of identical rank and shape.

The result shape/style follows the array operand (the left operand for array-array operations), and its value type is deduced
from applying the scalar operation. Two array operands must use the same style and memory space: `Array`/`Array_F`,
`SArray`/`SArray_F`, and host/device mixtures are rejected at compile time. No implicit index remapping or data migration is
performed. Dividing by zero and other scalar-domain behavior follow the underlying C++ operator.

### Unary operators and functions

The following accept a static or dynamic array and return a new same-shaped result with a deduced scalar type:

| Group | Operations |
| --- | --- |
| unary operators | `!a`, `+a`, `-a` |
| roots/magnitude | `abs(a)`, `sqrt(a)`, `cbrt(a)`, `pow(a, arithmetic_exponent)` |
| trigonometric | `sin`, `cos`, `tan`, `asin`, `acos`, `atan` |
| exponential/logarithmic | `exp`, `log`, `log10`, `log2` |
| rounding | `floor`, `ceil`, `round` |
| classification | `isnan`, `isinf` |

The namespace's `abs` is separate from `yakl::intrinsics::abs` but has similar array behavior. Math domain, range, NaN, and
infinity behavior come from the corresponding standard math operation on each component.

`componentwise::binary(left,right,functor)` and `componentwise::unary(value,functor)` are the generic implementation-facing
forms behind these operators. A device-callable scalar functor may be supplied to create a same-shaped result under the same
scalar/static/dynamic operand rules. `componentwise::same_shape` reports dynamic-array rank/extent compatibility.

## Performance and synchronization

Each dynamic componentwise call and each non-reduction dynamic intrinsic allocates a result and launches its own kernel.
Expressions are not fused. These APIs are particularly useful for testing, diagnostics, and concise transformations; a
single explicit `parallel_for` is normally preferable for a long production expression. Operations are asynchronous unless
`YAKL_AUTO_FENCE` or profiling introduces a fence. Keep all operands and returned allocations alive until queued work has
finished.

In debug builds, dynamic operands are checked for allocation and array-array shapes are checked before launch. The same
shape requirements remain mandatory in release builds.

[Previous: Parallel execution](parallel-execution.md) · [API home](README.md) · Next: [Random numbers and timers](random-timers.md)
