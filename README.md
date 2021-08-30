# YAKL: Yet Another Kernel Launcher
## A Simple C++ Framework for Performance Portability and Fortran Code Porting

Author: Matt Norman (Oak Ridge National Laboratory) - mrnorman.github.io

Contributors:
* Matt Norman (Oak Ridge National Laboratory)
* Isaac Lyngaas (Oak Ridge National Laboratory)
* Abhishek Bagusetty (Argonne National Laboratory)
* Mark Berrill (Oak Ridge National Laboratory)

**YAKL is still a work in progress, and the API is still in flux. Currently this only supports the E3SM-MMF Exascale Computing Project (ECP) application. It is not intended to compete with the more functional portability frameworks. YAKL has a focus on Fortran porting, simplicity, and readability.**

* [Overview](#overview)
* [Code Sample](#code-sample)
* [Using YAKL](#using-yakl)
  * [`parallel_for`](#parallel_for)
  * [`Array`](#array)
  * [Moving Data Between Two Memory Spaces](#moving-data-between-two-memory-spaces)
  * [`SArray` and `YAKL_INLINE`](#sarray-and-yakl_inline)
  * [`Array`, and `FSArray`, and Fortran-like `parallel_for` (Fortran Behavior in C++)](#fortran-style-array-fsarray-and-fortran-like-parallel_for-fortran-behavior-in-c)
  * [Managed Memory](#managed-memory)
  * [Pool Allocator](#pool-allocator)
  * [Synchronization](#synchronization)
  * [Atomics](#atomics)
  * [Reductions (Min, Max, and Sum)](#reductions-min-max-and-sum)
  * [ScalarLiveOut](#scalarliveout)
  * [Fortran - C++ Interoperability with YAKL: `Array` and `gator_mod.F90`](#fortran---c-interoperability-with-yakl-array-and-gator_modf90)
  * [Interoperating with Kokkos](#interoperating-with-kokkos)
* [Compiling with YAKL](#compiling-with-yakl)
* [YAKL Timers](#yakl-timers)
* [`YAKL SCOPE`](#yakl_scope)
* [Handling Class Methods Called from Kernels](#handling-class-methods-called-from-kernels)
* [A Note on Thread Safety](#a-note-on-thread-safety)
* [Future Work](#future-work)
* [Software Dependencies](#software-dependencies)

## Overview

The YAKL API is similar to Kokkos in many ways, but is quite simplified and has much stronger Fortran interoperability and Fortran-like options. YAKL currently has backends for CPUs (serial), CUDA, HIP, SYCL (in progress), and OpenMP offload (in progress). YAKL provides the following:

* **Pool Allocator**: A pool allocator for all `memDevice` `Array`s in accelerator memory
  * Supports `malloc`, `cudaMalloc`, `cudaMallocManaged`, `hipMalloc`, and `hipMallocHost` allocators
  * CUDA Managed memory also calls `cudaMemPrefetchAsync` on the entire pool
  * If the pool allocator is not used, YAKL still maintains internal host and device allocators with the afforementioned options
  * Specify `-DYAKL_MANAGED_MEMORY` to turn on `cudaMallocManaged` for Nvidia GPUs and `hipMallocHost` for AMD GPUs
  * Controllable via environment variables (initial size, grow size, and to turn it on or off)
* **Fortran Bindings**: Fortran bindings for the YAKL internal / pool device allocators
  * For `real(4)`, `real(8)`, `int(4)`, `int(8)`, and `logical` arrays up to seven dimensions
  * Using Fortran bindings for Managed memory makes porting from Fortran to C++ on the GPU significantly easier
  * You can call `gator_init()` from `gator_mod` to initialize YAKL's pool allocator from Fortran code.
  * Call `gator_finalize()` to deallocate YAKL's runtime pool and other miscellaneous data
  * When you specify `-DYAKL_MANAGED_MEMORY -DYAKL_ARCH_CUDA`, all `gator_allocate(...)` calls will not only use CUDA Managed Memory but will also map that data in OpenACC and OpenMP offload runtimes so that the data will be left alone in the offload runtimes and left for the CUDA runtime to manage instead.
    * This allows you to use OpenACC and OpenMP offload without any data statements and still get efficient runtime and correct results.
    * This is accomplished with `acc_map_data(...)` in OpenACC and `omp_target_associate_ptr(...)` in OpenMP 4.5+
    * With OpenACC, this happens automatically, but with OpenMP offload, you need to also specify `-D_OPEMP45`
* **Limited Fortran Intrinsics**: To help the code look more like Fortran when desired
  * A limited library of Fortran intrinsic functions (a reduced set, not all Fortran intrinsics)
  * Most intrinsics operator on Fortran-style and C-style arrays on the heap and on the stack
  * Some intrinsics (like `pack`) only operate in host memory
* **Multi-Dimensional Arrays**: A C-style `Array` class for allocated multi-dimensional array data
  * Only one memory layout is supported for simplicity: contiguous with the right-most index varying the fastest
    * This makes library interoperability more straightforward
  * Up to eight dimensions are supported
  * `Array`s support two memory spaces, `memHost` and `memDevice`, specified as a template parameter.
  * The `Array` API is similar to a simplified Kokkos `View`
  * `Array`s use shallow move and copy constructors (sharing the data pointer rather than copying the data) to allow them to work in C++ Lambdas via capture-by-value in a separate device memory address space.
  * `Array`s automatically use the YAKL pool allocator unless the user turns the pool off via an environment variable
  * `Array`s are by default "owned" (allocated, reference counted, and deallocated), but there are "non-owned" constructors that can wrap existing data pointers
  * Both C and Fortran style `Array`s allow simple array "slicing"
  * Supports array index debugging to throw an error when indices are out of bounds or the wrong number of dimensions is used
* **Multi-Dimensional Arrays On The Stack**: An `SArray` class for static arrays to be placed on the stack
  * This makes it easy to create low-overhead private arrays in kernels
  * Supports up to four dimensions, the sizes of which are template parameters known at compile time
    * CPU code allows runtime-length stack arrays, but most accelerators do not allow this
  * Supports array index debugging to throw an error when indices are out of bounds
  * Because `SArray` objects are inherently on the stack, they have no templated memory space specifiers
* **Fortran-style Multi-dimensional Arrays**: Fortran-style `Array` and `FSArray` classes
  * Left-most index varies the fastest just like in Fortran (still has a contiguous memory layout)
  * Arbitrary lower bounds that default to one but can vary to any integer
  * `parallel_for` kernel launchers that take arbitrary loop bounds and strides like Fortran loops
    * Lower bounds of loops default to one, and by default includes the upper bound in iterations (like Fortran `do` loops)
  * Contains a `slice()` function to allow simple contiguous Fortran-like array slices
  * Removes the need to permute array indices and change indexing when porting Fortran codes to C++
* **Kernel Launchers**: `parallel_for` launchers
  * Similar syntax as the Kokkos `parallel_for`
  * Only supports one level of parallelism for simplicity, so your loops need to be collapsed
    * Once you begin to expose multiple levels of on-node parallelism, you have inherently given up some ground in terms of portability
  * Multiple tightly-nested loops are supported through the `Bounds` class, which sets multiple looping indices for the kernel
  * C-style and Fortran-style `parallel_for` functions and `Bounds` classes
    * C-style defaults to: `parallel_for( nx , YAKL_LAMBDA (int i) {` --> `for (int i=0; i < nx; i++) {`
    * FOrtran-style defaults to: `parallel_for( nx , YAKL_LAMBDA (int i) {` --> `for (int i=1; i <= nx; i++) {`, which is the same as a Fortran `do`-loop
  * Supports CUDA, CPU serial, and HIP backends at the moment
  * `parallel_for` launchers on the device are by default asynchronous in the CUDA and HIP default streams
  * There is an automatic `fence()` option specified at compile time with `-DYAKL_AUTO_FENCE` to insert fences after every `parallel_for` launch
* **NetCDF I/O and Parallel NetCDF I/O**:
  * Simplified NetCDF reading and writing utilities to get dimension sizes, tell if dimensions and variables exist, and write data either as entire varaibles or as a single entry with an `unlimited` NetCDF dimension.
  * You can write an entire variable at once, or you can write just one entry in an unlimited (typically time) dimension
* **Atomics**: Atomic instructions
  * `atomicAdd`, `atomicMin`, and `atomicMax` are supported for 4- and 8-byte integers (signed and unsigned) and reals
  * Whenever possible, hardware atomics are used, and software atomics using `atomicCAS` and real-to-integer conversions in the CUDA and HIP backends
  * Note this is different from Kokkos, which specifies atomic accesses for entire Views all of the time. With YAKL, you only perform atomic accesses when you need them at the development expense of having to explicitly use `atomic[Add|Min|Max]` in the kernel code yourself.
* **Reductions**: Parallel reductions using the Nvidia `cub` and AMD `hipCUB` (wrapper to `rocPRIM`) libraries
  * Temporary data for a GPU device reduction is allocated with YAKL's internal device allocators and owned by a class for your convenience. It is automatically deallocated when the class falls out of scope. The user neither sees nor handles allocations needed for the library calls.
  * The user can create a class object and then reuse it to avoid multiple allocations during runtime
  * If the pool allocator is turned on, allocations are fast either way
  * The `operator(T *data)` function defaults to copying the result of the reduction to a host scalar value
  * The user can also use the `deviceReduce(T *data)` function to store the result into a device scalar location
* **Scalar Live-Out**: When a device kernel needs to return a scalar that depends on calculations in the kernel, the scalar has to be allocated in device memory. YAKL has a `ScalarLiveOut` class that makes this more convenient.
  * This is perhaps the most obscure among common issues encountered in GPU porting, but scalars that are assigned in a kernel and used outside the kernel must be allocated in device memory (perhaps using a 1-D YAKL `Array` of size 1), and the data must be copied from device to host once the kernel is done.
  * This situation happens most often with testing kernels; i.e., a `bool` decides if the data is valid or not, and it is read on the host outside the kernel.
  * `ScalarLiveOut` allows normal assignment with the `=` operator inside a kernel (so it looks like a normal scalar in the kernel).
  * `ScalarLiveOut` is initialized from the host via a constructor that allocates on the device and copies initial data the device behind the scenes
  * `ScalarLiveOut` allows reads on the host with a `hostRead()` member function that copies data from the device to the host behind the scenes.
* **Synchronization**
  * The `yakl::fence()` operation forces the host code to wait for all device code to complete

## Code Sample

The following loop would be ported to general accelerators with YAKL as follows:

```C++
#include "YAKL.h"
#include <iostream>
typedef float real;
typedef yakl::Array<real,4> real4d;
void applyTendencies(real4d &state2, real const c0, real4d const &state0,
                                     real const c1, real4d const &state1,
                                     real const ct, real4d const &tend,
                                     Domain const &dom) {
  real tot = 0;
  for (int l=0; l<numState; l++) {
    for (int k=0; k<dom.nz; k++) {
      for (int j=0; j<dom.ny; j++) {
        for (int i=0; i<dom.nx; i++) {
          state2(l,hs+k,hs+j,hs+i) = c0 * state0(l,hs+k,hs+j,hs+i) +
                                     c1 * state1(l,hs+k,hs+j,hs+i) +
                                     ct * dom.dt * tend(l,k,j,i);
          tot += state2(l,hs+k,hs+j,hs+i);
        }
      }
    }
  }
}
std::cout << state2;
std::cout << tot << std::endl;
```

will become:


```C++
#include "YAKL.h"
#include <iostream>
typedef float real;
typedef yakl::Array<real,4,yakl::memDevice> real4d;
void applyTendencies(real4d &state2, real const c0, real4d const &state0,
                                     real const c1, real4d const &state1,
                                     real const ct, real4d const &tend,
                                     Domain const &dom) {
  // for (int l=0; l<numState; l++) {
  //   for (int k=0; k<dom.nz; k++) {
  //     for (int j=0; j<dom.ny; j++) {
  //       for (int i=0; i<dom.nx; i++) {
  yakl::c::parallel_for( yakl::c::Bounds<4>(numState,dom.nz,dom.ny,dom.nx) ,
                         YAKL_LAMBDA (int l, int k, int j, int i) {
    state2(l,hs+k,hs+j,hs+i) = c0 * state0(l,hs+k,hs+j,hs+i) +
                               c1 * state1(l,hs+k,hs+j,hs+i) +
                               ct * dom.dt * tend(l,k,j,i);
  }); 
  real tot = yakl::intrinsics::sum( state2 );
}
std::cout << state2.createHostCopy();
std::cout << tot << std::endl;
```

## Using YAKL

To use YAKL's Array classes, `parallel_for` launchers, reductions, allocators, and atomics, you only need to `#include "YAKL.h"`. For NetCDF I/O routines, you need to `#include "YAKL_netcdf.h"`. For YAKL's FFTs, you need to `#include "YAKL_fft.h"`.

Be sure to use `yakl::init(...)` at the beginning of the program and `yakl::finalize()` at the end. If you do not call these functions, you will get errors during runtime for all `Array`s, `SArray`s, and device reductions.

### `parallel_for`

Preface functions you want to call from a device kernel with `YAKL_INLINE` or `YAKL_DEVICE_INLINE`. `YAKL_INLINE` functions can be called from the host or device, and `YAKL_DEVICE_INLINE` functions can only be called from the device. Preface "loop bodies" you're passing to YAKL launchers with `YAKL_LAMBDA` (which can run on the host or device) or `YAKL_DEVICE_LAMBDA` (which can only run on the device). The recommended use case for the `parallel_for` launcher is:

```C++
#include "YAKL.h"
...
// Non-standard loop bounds
// for (int j=0; j < ny; j++) {
//   for (int i=1; i <= nx-1; i++) {
yakl::parallel_for( Bounds<2>(ny,{1,nx-1}) , YAKL_LAMBDA (int j, int i) {
  ...
});

// Multiple tightly nested loops
// for (int j=0; j<ny; j++) {
//   for (int i=0; i<nx; i++) {
yakl::parallel_for( Bounds<2>(ny,nx) , YAKL_LAMBDA (int j, int i) {
  c(i,j) = a(i,j) * b(j);
});

// A single loop
// for (int i=0; i<nx; i++) {
yakl::parallel_for( nx , YAKL_LAMBDA (int i) {
  c(i) = a(i)*b(i);
});
```

The `parallel_for` launcher is useful for every type of loop except for prefix sums and reductions. You can embed atomics in `parallel_for` as you'll see a few sections down.

### `Array`

The `Array` class is set up to handle two different memories: Host and Device, and you can seen an example of how to use these above as well as in the [miniWeather](https://github.com/mrnorman/miniWeather) codebase.

The `Array` class can be owned or non-owned. The constructors are:

```C++
// Owned (Allocated, reference counted, and deallocated by YAKL)
yakl::Array<T type, int rank, int memSpace, int style>(char const *label, int dim1, [int dim2, ...]);

// Non-Owned (NOT allocated, reference counted, or deallocated)
// Use this to wrap existing contiguous data pointers (e.g., from Fortran)
yakl::Array<T type, int rank, int memSpace, int style>(char const *label, T *ptr, int dim1, [int dim2, ...]);
```

* Valid values for `T` can be any `std::is_arithmetic<T>` type in `memDevice` memory and any class at all in `memHost` memory. On the host, `Array`s use the C++ "placement `new`" approach to call the contained class's constructor. This is not supported in device memory because it is less portable, and it requires Unified memory between host and device. It is not recommended to use `Array` objects with a pointer type that has a constructor, as that constructor will not be called correctly on the device.
* Valid values for `rank` can be 1, 2, ..., 8
* Valid values for `memSpace` can be `memHost` or `memDevice` (default is `memDevice`)
* Valid values for `style` can be `styleC` or `styleFortran` (default is `styleC`)

Data is accessed in an `Array` object via the `(ind1[,ind2,...])` parenthesis operator with the right-most index varying the fastest for the C style `Array` (`yakl::styleC`) and the left-most index varying the fastest for the Fortran-style `Array` (`yakl::styleFortran`).

YAKL `Array` objects use shallow copies in the move and copy constructors. I.e., if you say `arr1 = arr2`, these two objects now **share the same data pointer**, and only the metadata is actually copied over. To copy the actual data from one `Array` to another, use the `Array::deep_copy_to(Array &destination)` member function. This will copy data between different memory spaces (e.g., `cudaMemcpy(...)`) for you depending on the memory spaces of the `Array` objects.

To create a host copy of an `Array`, use `Array::createHostCopy()`; and to create a device copy, use `Array::createDeviceCopy()`.

* `C` behavior only supports lower bounds of `0` for each dimension, and it has the right-most index varying the fastest in memory.
* `Fortran` behavior allows arbitrary lower bounds that defualt to `1`, and it has the left-most index varying the fastest in memory.

Both styles of `Array` only support data that is contiguous in memory.

### Moving Data Between Two Memory Spaces

The intent of YAKL is that you will want copies of an `Array` object in one of two distinct memory spaces: Host (i.e., main memory) and Device (e.g., GPU memory). There are currently four member functions of the `Array` class to help with data movement:

```C++
// Create a copy of this Array class in Host Memory, and pass that copy back as a return value.
Array<...> createHostCopy();

// Create a copy of this Array class in Device Memory, and pass that copy back as a return value.
Array<...> createDeviceCopy();

// Copy the data from this Array pointer to the Host Array's pointer (Host Array must already exist)
void deep_copy_to(Array<T,memHost> lhs);

// Copy the data from this Array pointer to the Device Array's pointer (Device Array must already exist)
void deep_copy_to(Array<T,memDevice> lhs);
```

### `SArray` (and `YAKL_INLINE`)

To declare local data on the stack inside a kernel or stack data on the host that will be copied by value to the device, you'll use the `SArray` class for C-like behavior and the `FSArray` class for Fortran-like behavior. For instance:

```C++
typedef float real;
yakl::Array<real,memDevice> state("state",nx+2*hs);
yakl::Array<real,memDevice> coefs("coefs",nx,ord);

...

YAKL_INLINE void recon( SArray<real,ord> const &stencil , 
                        SArray<real,ord> &coefs ) {
  ...
}
                        
yakl::parallel_for( nx , YAKL_LAMBDA (int i) {
  // stencil and coefs are defined on the stack inside the kernel
  yakl::SArray<real,1,ord> stencil, coefsLoc;
  for (int ii=0; ii<ord; ii++) { stencil(ii) = state(i+ii); }
  // recon is callable from within a kernel because of the YAKL_INLINE prefix
  recon(stencil,coefsLoc);
  for (int ii=0; ii<ord; ii++) { coefs(i,ii) = coefsLoc(ii); }
  ...
});
```

`SArray` uses the following form:

```C++
SArray<type , num_dimensions , dim1 [ , dim2 , ...]> arrName;
```

* `type` should be kept to `std::is_arithmetic` types, though you can use classes with constructors at your own risk.
* `num_dimensions` is 1, 2, 3, or 4
* Number of dimensions provided after `num_dimensions` must match `num_dimensions`, or a compiler-time error will result.

Data is accessed in an `SArray` object via the `(ind1[,ind2,...])` parenthesis operator with the right-most index varying the fastest.

Note that `SArray` objects are inherently placed on the stack of whatever context they are declared in. Because of this, it doesn't make sense to allow a templated memory specifier (i.e., `yakl::memHost` or `yakl::memDevice`). Therefore, stack `Array` objects always have the `yakl::memStack` specifier. If used in a CUDA kernel, `SArray` objects will be placed onto the kernel stack frame. If used in CPU functions, they will be placed in the CPU function's stack. If defined on the host and used in a kernel, it will be copied by value to the device. Large `SArray` transfers from host to device will incur a large transfer cost.

## Fortran-style `Array`, `FSArray`, and Fortran-like `parallel_for` (Fortran Behavior in C++)

YAKL also has a Fortran-style `Array` and `FSArray` classes to make porting Fortran code to C++ simpler. It's time-consuming and error-prone to have to reorder the indices of every array to row-major and change all indexing to match C's always-zero lower bounds. Also, Fortran allows array slicing and non-one lower bounds, which can make porting very tedious. Look-up tables and intermediate index arrays and functions make this even harder.

The Fortran-style `Array` and `FSArray` classes allow any arbritrary lower bound, defaulting to one, and the left-most index always varies the fastest like in Fortran. You can create them as follows:

**Fortran-style `Array`**:
* `Array<float,2,yakl::memHost,yakl::styleFortran> arr( "label" , int dim1 , int dim2 )`
  * Equivalent to Fortran: `real, allocatable :: arr(:,:); allocate(arr(dim1,dim2))`
  * Creates an owned array
* `Array<float,1,yakl::memHost,yakl::styleFortran> arr( "label" , float *data_p , int dim1 )`
  * Equivalent to Fortran: `real, pointer :: arr(:); arr(:)->data(:)`
  * Same as before but non-owned (wraps an existing contiguous data pointer)
* `Array<double,2,yakl::memHost,yakl::styleFortran> arr( "label" , {-1,52} , {0,43} )`
  * Equivalent to Fortran: `real(8), allocatable :: arr(:,:); allocate(arr(-1:52 , 0:43)`
  * Lower and upper bounds are specified as C++ integer initializer lists
* `Array<double,2,yakl::memHost,yakl::styleFortran> arr( "label" , double *data_p , {-1,52} , {0,43} )`
  * Equivalent to Fortran: `real(8), pointer :: arr(:,:); arr(-1:,0:) -> data(:,:)`
  * Same as before but non-owned (wraps an existing contiguous data pointer)
* **slice()**: `using yakl::COLON;  Array... arr;  arr.slice<2>(COLON,COLON,ind1,ind2)`
  * Equivalent to Fortran array slicing: `arr(:,:,ind1,ind2)`
  * Only works on simple, *contiguous* array slices with *entire dimensions* (not partial dimensions) sliced out
    * E.g., `arr(0:5,4,7)`, though contiguous is not supported
  * If you want to **write** to the array slice passed to a function, you must save it as a temporary variable first and pass the temporary variable: E.g., `auto tmp = arr.slice<2>({COLON,COLON,ind1,ind2});  myfunc(tmp);`
    * This is because C++ treats objects returned by value as `const`
  * If you're only reading from the array slice, you can pass it directly inline because it obeys `const`
  * `slice()` always produces non-owned Fortran-style Arrays of the same type in the same memory space wrapping a contiguous portion of the host `Array`

**`FSArray`**:
`FSArray` uses the following form:

```C++
// Specify lower and upper bounds
FSArray<type , num_dimensions , SB<lb_1,ub_1> [ , SB<lb_2,ub_2> , ...]> arrName;

// Specify only upper bounds
FSArray<type , num_dimensions , SB<ub_1> [ , SB<ub_2> , ...]> arrName;

// Or specify some bounds as SB<lb,ub> and others as only SB<ub> in the same FSArray object
```

While the `SB<>` syntax admittedly looks messy, the `SB<lower_bound,upper_bound>` (or Static Bound) class is required to allow `FSArray` objects to have non-1 lower bounds, and C++ does not allow you to do this with prettier looking initializer lists like `{lb,ub}`. 

* Must template on the lower and upper bounds to place it on the stack.
* `FSArray<float,SB<-hs,hs>> stencil;`
  * Equivalent to the Fortran "automatic" array: `real :: stencil(-hs:hs)`
  * Low-overhead, placed on the stack, inherently thread-private in kernels

**`parallel_for` (Fortran style)**:
```C++
using yakl::Bounds;
yakl::parallel_for( Bounds<2>({-1,30,3},{0,29}) , YAKL_LAMBDA ( int j, int i ) {
  // Loop body
});
```

This is the parallel C++ equivalent of the following Fortran code:

```fortran
do j=-1,30,3
  do i=0,29
    ! Loop body
  enddo
enddo
```

Currently, backwards iterating is not supported, so `do` loops with a negative stride will need to be altered to have a positive stride.

**Fortran Intrinsics**:

There are a number of Fortran intrinsic functions available that can operate on Fortran-like `Array`s or `FSArray`s. For instance:

```C++
Array<float,2,memDevice,styleFortran> arr("arr",5,5);
std::cout << shape(arr);
std::cout << size(arr,2);
std::cout << sum(arr);
if (allocated(arr)) { ... }
if (associated(arr)) { ... }
// etc.
```

### Managed Memory

To use CUDA Managed Memory or HIP pinned memory, add `-DYAKL_MANAGED_MEMORY` to the compiler flags. This will make your life much easier when porting a Fortran code to a GPU-enabled C++ code because all data allocated with the Fortran hooks of YAKL will be avilable on the CPU and the GPU without having to transfer it explicitly. However, it is always recommended to handle the transfers yourself eventually for improved efficiency. 

### Pool Allocator

An easy-to-use, self-contained, automatically growing C++ pool allocator optimized for stack-like allocations and deallocations with fortran bindings and hooks into OpenMP offload and OpenACC, CUDA Managed memory, and HIP.

```
   ,(   ,(   ,(   ,(   ,(   ,(   ,(   ,(
`-'  `-'  `-'  `-'  `-'  `-'  `-'  `-'  `
   _________________________
 / "Don't be a malloc-hater  \
|   Use the pool alligator!"  |
 \     _____________________ / 
  |  /
  |/       .-._   _ _ _ _ _ _ _ _
.-''-.__.-'00  '-' ' ' ' ' ' ' ' '-.
'.___ '    .   .--_'-' '-' '-' _'-' '._
 V: V 'vv-'   '_   '.       .'  _..' '.'.
   '=.____.=_.--'   :_.__.__:_   '.   : :
           (((____.-'        '-.  /   : :
                             (((-'\ .' /
                           _____..'  .'
                          '-._____.-'
   ,(   ,(   ,(   ,(   ,(   ,(   ,(   ,(
`-'  `-'  `-'  `-'  `-'  `-'  `-'  `-'  `
```

**Author:** Matt Norman, Oak Ridge National Laboratory

#### Features

* Fortran bindings for integer, integer(8), real, real(8), and logical
* Fortran bindings for arrays of one to seven dimensions
* Able to call cudaMallocManaged under the hood with prefetching and memset
* Able to support arbitrary lower bounds for Fortran allocations
* Simple and efficient pool allocator implementation that's easy to compile and automatically grows as needed
* The pool allocator responds to environment variables to control the initial allocation size, growth size, and block size
* No segmentation for stack-like allocations and deallocations for efficient use of limited memory space.
* Warns the user if allocations are left allocated after the pool is destroyed to help the user debug the infamous "double free" error.

#### Why Use A Memory Pool?

In most of our codes in weather and climate, the reason we need a pool allocator on GPUs is that the native `cudaMalloc` and `hipMalloc` for Nvidia and AMD GPUs (presumably Intel will fall into this category as well) have extremely large runtimes that scale with the size of the allocation. This is for two reasons: (1) `cudaMalloc` is basically a system-level call, and (2) there is no Operating System layer on GPUs to manage a pool for us like we have in the Linux kernel in main system memory. Further, `cudaFree` and `hipFree` are synchronizing calls because they cannot deallocate memory while kernels are still using that memory. These issues combined make frequent intermittent malloc and free operations prohibitively expensive on GPUs, and the speed-up is usually reduced by at least 5x in our strong-scaled codes.

#### What Does "Stack-Like" Allocations and Deallocations Mean?

In weather and climate, we tend to allocate local data dynamically at the beginning of subroutines and then deallocate that data at the end of the subroutines. It ends up lookng like this:

```Fortran
subroutine parent()
   allocate(parent_a(...))
   allocate(parent_b(...))
   call child1()
   call child2()
   ! do work
   deallocate(parent_a))
   deallocate(parent_b))
end subroutine parent

subroutine child1()
   allocate(child1_data(...))
   ! do work
   deallocate(child1_data)
end subroutine child1

subroutine child2()
   allocate(child2_data(...))
   ! do work
   call grandchild()
   deallocate(child2_data)
end subroutine child2

subroutine grandchild()
   allocate(grandchild_data(...))
   ! do work
   deallocate(grandchild_data)
end subroutine randchild
```

If you follow the order of allocations and deallocations in that code, it's very close to a stack:

* `allocate(parent_a(...))`
* `allocate(parent_b(...))`
  * `allocate(child1_data(...))`
  * `deallocate(child1_data)`
  * `allocate(child2_data(...))`
    * `allocate(grandchild_data(...))`
    * `deallocate(grandchild_data)`
  * `deallocate(child2_data)`
* `deallocate(parent_a))`
* `deallocate(parent_b))`

You push allocations onto the back of the stack, and then your free allocations from the back of the stack (First In, Last Out). The most efficient memory allocator in terms of space efficiency and speed is, in fact, a stack. However, a stack **requires** you to deallocate in the reverse order that you allocate, and it can't be thread-safe. A stack-like allocator is more flexible. It allows you to allocate and deallocate in any order you want, and it can be made thread safe.

However, unlike a more general allocator like a Buddy Allocator (https://en.wikipedia.org/wiki/Buddy_memory_allocation), the stack-like allocator here assumes the allocations and deallocations are **close** to stack-like behavior, and it optimizes for that case. In essence, new allocations can only be pushed to the end of the allocation list (there is no searching for gaps between previous allocations in the list), and when a pointer is deallocated, the search for the allocation for that pointer begins at the end of the allocation list and traverses backward.

With this assumption, any violation of stack-like behavior will incur linear search costs and additional segmentation. But for most of weather and climate, the behavior is very close to that of a stack.

#### Automatically Growing

Gator automatically allocates more space whenever it runs out of space for a requested allocation. This is done by adding additional pools to a list. The behavior is still the same as it was with a single pool in terms of allocation and deallcation searching. The new pools will remain in place for the duration of the simulation. Therefore, once a pool has grown, it does not need to grow again until the memory requirements reach a new memory high-water mark, which they rarely do in weather and climate after the first time step.

#### Informative

Gator keeps track of the memory high-water mark in the pool automatically and informs the user at the end of the simulation.

Error messages are made as helpful as possible to help the user know what to do in the event of a pool overflow. Sometimes, it is necessary to create a larger initial pool to use a limited memory space more effectively, and this guidance is given in error messages as errors occur.

Also, all allocations can be labeled with an optional string parameter upon allocation. This allows tracking of timestamped and labeled allocations that are written to a file in debug mode (**soon to be implemented**).

#### Environment variables 

You control the behavior of Gator's pool management through the following environment variables:

* `GATOR_INITIAL_MB`: The initial pool size in MB
* `GATOR_GROW_MB`: The size of each new pool in MB once the initial pool is out of memory
* `GATOR_BLOCK_BYTES`: The size of a byte. I can't imagine why you'd want to change this, but you can

Environment variables are advantageous because they allow easier adaptation to critical metrics like the number of processes per node or per GPU, which are easiest to calculate outside of the executable itself.

#### GPUs and Managed Memory

The following CPP defines control Gator's behavior as well:

* `-DYAKL_ARCH_CUDA`: Enable CUDA allcoations (`cudaMalloc` and `cudaFree` are used to create and destroy the pools)
* `-DYAKL_ARCH_HIP`: Enable HIP allocations (`hipMalloc` and `hipFree` are used to create and destroy the pools)
* `-DYAKL_MANAGED_MEMORY`: Enable managed memory (`cudaMallocManaged` and `hipMallocHost` are used for CUDA and HIP, respectively, and the pools are pre-fetched to the GPU ahead of time for you), and inform OpenMP and OpenACC runtimes of this memory's Managed status whenever OpenACC or OpenMP are enabled. This is done automatically for OpenACC, but for OpenMP45, you'll need to specify a CPP define
* `-D_OPENMP45 -DYAKL_MANAGED_MEMORY`: Tell the OpenMP4.5 runtime that your allocations are managed so that OpenMP doesn't try to copy the data for you (i.e., this lets the underlying CUDA runtime handle it for you instead). This only does anything if `-DYAKL_MANAGED_MEMORY` is also specified.

### Synchronization

Currently YAKL only supports asynchronicity through the default CUDA stream. To synchronize the host code with code running on a GPU device, use the `yakl::fence()` functions which no-ops in host code and calls `cudaDeviceSynchronize()` on Nvidia GPUs and `hipDeviceSynchronize()` on AMD GPUs.

### Atomics

When you need atomic operations in YAKL, you can use the `atomicAdd`, `atomicMin`, and `atomicMax` routines. The most common use-case for atomics is some form of partial reduction over some of the indices of an array but not all. For example:

```C++
yakl::Array<real,4,memDevice> :: arrLarge("arrLarge",nz,ny,nx,ncrm);
yakl::Array<real,2,memDevice> :: arrSmall("arrSmall",nz,ncrm);

...

// for (int k=0; k<nzm; k++) {
//   for (int j=0; j<ny; j++) {
//     for (int i=0; i<nx; i++) {
//       for (int icrm=0; icrm<ncrms; icrm++) {
yakl::parallel_for( Bounds<4>(nzm,ny,nx,ncrms) , YAKL_LAMBDA (int k, int j, int i, int icrm) {
  // The operation below is a race condition in parallel. It needs an atomicMax
  // arrSmall(k,icrm) = max( arrSmall(k,icrm) , arrLarge(k,j,i,icrm) );
  yakl::atomicMax( arrSmall(k,icrm) , arrLarge(k,j,i,icrm) );
});
```

As a rule, if you ever see anything on the left-hand-side of an `=` with **fewer indices than you have surrounding loops**, then you're going to have a race condition that requires an atomic access.

### Reductions (Min, Max, and Sum)

The best way to do reductions is through the YAKL intrinsic functions `sum`, `maxval`, and `minval`; which each take a single `Array` parameter of any memory space, style, type, or rank. E.g., `minval(arr)`, where `arr` is an `Array` on the device will perform a minimum reduction of the data efficiently on the accelerator device and then pass the result back to the host. The routines below are available for performing reductions on general contiguous pointers of data OR if you need to keep the result of the reduction on the device and avoid the copy of the result back to the host.

YAKL provides efficient min, max, and sum array reductions using [CUB](https://nvlabs.github.io/cub/) and [hipCUB](https://github.com/ROCmSoftwarePlatform/hipCUB) for Nvidia and AMD GPUs. Because these implementations require temporary storage, a design choice was made to expose reductions through class objects. Upon construction, you must specify the size (number of elements to reduce), type (`template <class T>`) of the array that will be reduced, and the memory space (via template parameter, `yakl::memHost` or `yakl::memDevice`) of the array to be reduced. The constructor then allocates memory for the temporary storage. Then, you run the reduction on an array of that size using `T operator()(T *data)`, which returns the result of the reduction in host memory. When the object goes out of scope, it deallocates the data for you. The array reduction objects are not sharable and implements no shallow copy. An example reduction is below:

```C++
Array<float> dt3d;
// Fill dt3d
yakl::ParallelMin<float,yakl::memDevice> pmin( nx*ny*nz );
dt = pmin( dt3d.data() );
```

If you want to avoid copying the result back to the host, you can run the `void deviceReduce(T *data, T *rslt)` member function, where the `rslt` pointer is allocated in device memory. An example is below:

```C++
Array<float> dt3d;
float *dtDev;
// Allocate dtDev on device
// Fill dt3d
yakl::ParallelMin<float,yakl::memDevice> pmin( nx*ny*nz );
pmin.deviceReduce( dt3d.data() , dtDev );
```

As a rule, if you ever see a scalar on the left-hand and right-hand sides of an `=`, then it's a race condition in parallel that you will need to resolve by using a reduction.

### `ScalarLiveOut`

When you write to a scalar in a device kernel and need to subsequently read that value on the host, you encounter a "scalar live-out" scenario, and some compilers even tell you when this happens (though some do not). This happens most often in the following scenario:

* __Testing routines__: You pass through some data and determine whether it's realistic or not, assigning this to a `bool` that is read on the host later to report the error.

These situations are reductions in nature, but often it's not convenient or efficient to express them as reductions. 

In these cases, the scalar must be explicitly allocated in device memory, the initial scalar value transferred from host to device memory, the scalar value altered in the kernel, and the scalar value transferred from device to host memory after the kernel. `ScalarLiveOut` handles all of this for you as follows:

```C++
// Creates a bool scalar that is allocated in device memory
// and has an initial value of false (which is transferred
// to device memory for you in the constructor)
yakl::ScalarLiveOut<bool> dataIsBad(false);

yakl::c::parallel_for( yakl::c::Bounds<2>(ny,nx) ,
                       YAKL_LAMBDA (int j, int i) {
  // The ScalarLiveOut class overloads operator=, so you can
  // simply assign to it like any other scalar inside a kernel
  if (density(j,i) < 0 || pressure(j,i) < 0) {
    dataIsBad = true;
  }
});

// To read on the host after a kernel, use the hostRead()
// member function, which transfers the value to the host for you
if (dataIsBad.hostRead()) {
  std::cout << "ERROR: Invalid density or pressure!\n";
  throw ...
}
```

**When to not use `ScalarLiveOut`:** If you find yourself wanting to use atomics on a scalar, often times you're better off using a reduction instead, because all of the data is being reduced to a single scalar value. To facilitate this, it's best to create a temporary array with all necessary calculations (e.g., `dt3d` for the stable time step at each cell in a 3-D grid), and then perform a reduction on that array. While there is an `operator()` to expose the scalar for reading on the GPU, if you're needing to do this, there is usually an easier solution to you problem.

### Fortran - C++ interoperability with YAKL: `Array` and `gator_mod.F90`

We provide the `gator_mod` Fortran module to interface with YAKL's internal device allocator. To use it, you'll first need to make all "automatic" fortran arrays into `allocatable` arrays and explicitly allocate them:

```Fortran
real var1( lbnd_x:ubnd_x , lbnd_y:ubnd_y , lbnd_z:ubnd_z )
real var2(nx,ny,nz)
```

will become: 

```Fortran
real, allocatable :: var1(:,:,:)
real, allocatable :: var2(:,:,:)
...
allocate(var1( lbnd_x:ubnd_x , lbnd_y:ubnd_y , lbnd_z:ubnd_z ))
allocate(var2(nx,ny,nz))
...
deallocate(var1)
deallocate(var2)
```

Next, to interface with `gator_mod`, you'll need to transform all `allocatables` into `pointer, contiguous`. The `contiguous` attribute is recommended because some compilers will perform more optimizations on Fortran pointers when it is present. The resulting code is:

```Fortran
use gator_mod, only: gator_allocate, gator_deallocate

real, pointer, contiguous :: var1(:,:,:)
real, pointer, contiguous :: var2(:,:,:)
...
call gator_allocate( var1 , (/ ubnd_x-lbnd_x+1 , ubnd_y-lbnd_y+1 , ubnd_z-lbnd_z+1 /) , &
                            (/ lbnd_x , lbnd_y , lbnd_z /) )
call gator_allocate( var2 , (/nx,ny,nz/) )
...
call gator_deallocate(var1)
call gator_deallocate(var2)
```

Note that YAKL **does** support non-1 lower bounds in Fortran in `gator_allocate()`. The first set of indices specify the total extents of each dimension. The second set of indices specify the lower bounds of each dimension in case they aren't the default 1 lower bounds of Fortran. No, it's not the most convenient looking syntax for the user, but it is easier to implement this way :).

If a Fortran routine uses module data, when porting to C++, it is often easiest to mirror the Fortran practice. In this case, it is best to pass to allocate the Fortran data with `gator_allocate` and then immediately pass it to a C++ wrapping function, which will wrap it in unmanaged (non-owned) YAKL `Array` classes. With the above data, you will have the following code to do that:

```Fortran
module cpp_interface_mod
  use iso_c_binding
  interface
    subroutine wrap_arrays(var1,var2) bind(C,name="wrap_arrays")
      implicit none
      real(c_float), dimension(*) :: var1, var2
    end subroutine wrap_arrays
  end interface
contains
  ! All scalars can be directly bound to C
  real(c_float) , bind(C) :: scalar1, scalar2, scalar3
  
  ! Parameters cannot use bind(C), but rather must be replicated in
  ! a C++ header file with the constexpr keyword
  integer(c_int), parameter :: nx=DOM_NX, ny=DOM_NY, nz=DOM_NZ
  integer(c_int), parameter :: lbnd_x=-1, ubnd_x=nx+2
  integer(c_int), parameter :: lbnd_y=-1, ubnd_y=ny+2
  integer(c_int), parameter :: lbnd_z=-1, ubnd_z=nz+2
  
  ! Pass all arrays to a wrapper function to wrap them in Fortran-style Array objects in C++
  ! It's best to do this even for automatic arrays as well
  real(c_float) :: var1( lbnd_x:ubnd_x , lbnd_y:ubnd_y , lbnd_z:ubnd_z )
  real(c_float) :: var2( nx , ny , nz )
end module cpp_interface_mod
```

Fortran assumed size arrays (i.e., `dimension(*)`) are very convenient because an array of any dimension can be legally passed to the subroutine. Since Fortran passes by reference by default, the location of the data is passed to this array.

All variables in Fortran modules need to be moved to `iso_c_binding` types such as: `real(c_float)`, `real(c_double)`, `integer(c_int)`, and `logical(c_bool)`. All module-level scalars can directly be bound to C with `bind(C)`. Arrays, however, need to be passed to a wrapping routine.

In C++, you would have a header file, `fortran_data.h` with the following:

```C++
#pragma once

int constexpr nx=DOM_NZ;
int constexpr ny=DOM_NZ;
int constexpr nz=DOM_NZ;
int constexpr lbnd_x=-1; 
int constexpr lbnd_y=-1; 
int constexpr lbnd_z=-1; 
int constexpr ubnd_x=nx+2; 
int constexpr ubnd_y=ny+2; 
int constexpr ubnd_z=nz+2; 

typedef yakl::Array<float,3,yakl::memDevice,yakl::styleFortran> real3d;

extern "C" void wrap_arrays(float *var1, float *var2);

// Declare Array wrappers defined in fortran_data.cpp
extern real3d var1, var2;

// Declare external scalars defined in Fortran code
extern int scalar1, scalar2, scalar3;  

#endif
```

And you would have a source file, `fortran_data.cpp`, with:

```C++
#include "fortran_data.h"

real3d var1, var2;

extern "C" void wrap_arrays(float *var1_p, float *var2_p) {
  // These create un-owned YAKL Arrays using existing allocations from Fortran
  var1 = real3d( "var1" , var1_p , {lbnd_x:ubnd_x} , {lbnd_y:ubnd_y} , {lbnd_x:ubnd_x} );
  var2 = real3d( "var2" , var2_p , nx, ny, nz   );
}
```

Notice that because of the YAKL Fortran-style `Array` class, you do not need to change the way you index these arrays in the C++ code at all. It's column-major ordering like Fortran, it defaults to lower bounds of 1, and it supports non-1 lower bounds as well. In the end, you often have C++ code that looks nearly identical to your previous Fortran code, with the exception of any Fortran intrinsics not supported by YAKL.

Any time Fortran data is passed by parameter, you can use the un-owned `Array` constructors to wrap them just as seen above in the `wrap_arrays` functions. 

We've already seen how to pass Fortran arrays to C++. For scalars, though, consider the following Fortran function:

```fortran
module mymod
contains
  function blah(n,x,which,y) result(z)
    integer(c_int) , intent(in   ) :: n
    real(c_float)  , intent(in   ) :: x
    logical(c_bool), intent(in   ) :: which
    real(c_double) , intent(  out) :: y
    real(c_float)                  :: z
    ! code
    ! y = ...
  end function
end module
```

When you port this to C++, you'll change the Fortran code to the following:

```fortran
module mymod
  interface 
    function blah(n,x,which,y) result(z)  bind(C, name="blah")
      integer(c_int) , intent(in   ), value :: n
      real(c_float)  , intent(in   ), value :: x
      logical(c_bool), intent(in   ), value :: which
      real(c_double) , intent(  out)        :: y
      real(c_float)                         :: z
    end function
  end interface
contains
  ! You remove the code from here
  ! And you only put the function *header* above (no code)
end module
```

And you'll have a C++ function:

```C++
extern "C" float blah( int n , real x , bool which , double &y ) {
  // code
  // real y = ...
  return y;
}
```

Notice that in the Fortran interface, any scalar with `intent(in)` must be passed by value and not by reference (i.e., the `value` fortran keyword must be added). However, since the scalar `y` has `intent(out)`, we must pass it by reference, meaning we do not use the `value` keyword. On the C++ side of things, we accept `n`, `x`, and `which` as a simple `int`, `real`, and `bool`. But we accept `y` as `&y`, meaning when we change its value, the change lives outside the function like we want it to.

Regarding return values for functions, do not return a reference like you do with dummy arguments. If you added the `&` to the function return, you would essentially be returning a reference to a locally scoped variable, which won't have any meaning outside the function. The `iso_c_binding` handles things appropriately for you, so just return the appropriate type by value in C++, and map it to a simple return of the same type in the Fortran `interface` block.

The following table can help you convert Fortran parameter types to Fortran interfaces, and C++ dummy arguments:

| Fortran datatype                  | Fortran interface              | C++ dummy argument     |
| --------------------------------- | ------------------------------ | ---------------------- |
| `integer(c_int), intent(in)`      | `integer(c_int), value`        | `int`                  |
| `real(c_double), intent(out)`     | `real(c_double)`               | `double &`             |
| `real(c_float), dimension(...)`   | `real(c_float), dimension(*)`  | `float *`              |
| `logical(c_bool), dimension(...)` | `logical(c_bool), dimension(*)`| `bool *`               |


### Interoperating with Kokkos

YAKL `Array` objects can wrap a Kokkos `View` pointer the same way unmanaged Kokkos `View` constructors wrap existing pointers. To explain Kokkos-YAKL interoperability in an intuitive manner, it is easiest to demonstrate it with the following loops, which are all equivalent:

```C++
{ // Kokkos LayoutRight
  // YAKL can only directly interact with explicitly LayoutRight or LayoutLeft Kokkos View objects
  View<double **, LayoutRight, CudaSpace> view_dev   ("view_dev" ,ny,nx);
  View<double **, LayoutRight, HostSpace> view_host  ("view_host",ny,nx);
  // C-style YAKL arrays are indexed identically to Kokkos LayoutRight Views
  Array<double,2,memDevice,styleC>       array_dev_c ("array_dev_c" ,view_dev .data(),ny,nx);
  // Fortran-style YAKL arrays have indices permuted compated to Kokkos LayoutRight Views
  Array<double,2,memDevice,styleFortran> array_dev_f ("array_dev_f" ,view_dev .data(),nx,ny);
  Array<double,2,memDevice,styleFortran> array_dev_f2("array_dev_f" ,view_dev .data(),{2,nx+1},{-1,ny-2});
  Array<double,2,memHost,styleC>         array_host_c("array_host_c",view_host.data(),ny,nx);

  // C-style for loops go from 0,bnd-1
  // The implied loop ordering is always right-most index is the innermost loop
  // for (int j=0; j < ny; j++) {
  //   for (int i=0; i < nx; i++) {
  yakl::c::parallel_for( yakl::c::Bounds<2>(ny,nx) , YAKL_LAMBDA (int j, int i) {
    // C's fastest varying index is the right-most
    view_dev   (j,i) = 0;
    array_dev_c(j,i) = 0;
  });

  // Fortran-style for loops go (by default) from 1,bnd
  // The implied loop ordering is always right-most index is the innermost loop
  // do j = 1 , ny
  //   do i = 1 , nx
  yakl::fortran::parallel_for( yakl::fortran::Bounds<2>(ny,nx) , YAKL_LAMBDA (int j, int i) {
    // Fortran's fastest varying index is the left-most
    array_dev_f(i,j) = 0;
  });

  // This is an example of a Fortran-style loop nest with non-standard loop bounds
  // do j = -1 , ny-2
  //   do i = 2 , nx+1
  yakl::fortran::parallel_for( yakl::fortran::Bounds<2>({-1,ny-2},{2,nx+1}) , YAKL_LAMBDA (int j, int i) {
    // Fortran's fastest varying index is the left-most
    array_dev_f2(i,j) = 0;
  });

  for (int j=0; j < ny; j++) {
    for (int i=0; i < nx; i++) {
      view_host   (j,i) = 0;
      array_host_c(j,i) = 0;
    }
  }

  // If we wanted all of the device arrays in the *same* parallel_for, then the equivalent lines are
  // as follows, though I'm not sure you'd ever do this in practice
  // for (int j=0; j < ny; j++) {
  //   for (int i=0; i < nx; i++) {
  yakl::c::parallel_for( yakl::c::Bounds<2>(ny,nx) , YAKL_LAMBDA (int j, int i) {
    // C's fastest varying index is the right-most
    view_dev    (j,i) = 0;
    array_dev_c (j,i) = 0;
    array_dev_f (i+1,j+1) = 0;
    array_dev_f2(i+2,j-1) = 0;
  });
} // Kokkos LayoutRight

{ // Kokkos LayoutLeft
  // Kokkos's LayoutLeft View means that its left-most index is varying the fastest
  // just like Fortran's
  View<double **, LayoutLeft, CudaSpace> view_left_dev("view_left_dev" ,nx,ny);
  Array<double,2,memDevice,styleC>       array_dev_c ("array_dev_c" ,view_left_dev.data(),ny,nx);
  Array<double,2,memDevice,styleFortran> array_dev_f ("array_dev_f" ,view_left_dev.data(),nx,ny);

  // do j = 1 , ny
  //   do i = 1 , nx
  yakl::fortran::parallel_for( yakl::fortran::Bounds<2>(ny,nx) , YAKL_LAMBDA (int j, int i) {
    // Fortran's fastest varying index is the left-most
    view_left_dev(i,j) = 0;
    array_dev_f  (i,j) = 0;
  });

  // for (int j=0; j < ny; j++) {
  //   for (int i=0; i < nx; i++) {
  yakl::c::parallel_for( yakl::c::Bounds<2>(ny,nx) , YAKL_LAMBDA (int j, int i) {
    // C's fastest varying index is the right-most
    array_dev_c(j,i) = 0;
  });
} // Kokkos LayoutLeft
```

Further, with the above example, the indexed arrays `view_dev(0,3)`, `array_dev_c(0,3)`, and `array_dev_f(4,1)` all point to the same memory location.

In all of the cases, each `View` and `Array` object is simply wrapping an allocation of contiguous memory where some dimension (`nx` in this case) is varying the fastest. Therefore, you can simply wrap the pointers in different objects and then index appropriately. The Fortran-style YAKL `Array` object is by default indexed starting at one. Therefore, the Fortran-style `Array` object (just like the Fortran compiler) internally subtracts the lower bound from the index before computing offsets into the contiguous allocated data. For instance, a Kokkos 1-D `View` at index zero is the same memory as a YAKL Fortran-style (with default lower bounds) `Array` at index 1.
  
Both Kokkos and YAKL have unmanaged / un-owned multi-dimensional arrays, so you can wrap equivalent types using the data pointer, which each expose via `Array::data()` and `View::data()`

YAKL `parallel_for` launchers can use Kokkos `Views` without issue, and Kokkos `parallel_for` and `parallel_reduce` launchers can use YAKL `Array` objects without issue.

You can use YAKL atomic functions inside Kokkos `parallel_for` launchers. 

You can use YAKL `SArray` and `FSArray` objects inside Kokkos `parallel_for` launchers.

YAKL and Kokkos `fence()` operations are pretty much equivalent and can be used in either framework.

YAKL's `YAKL_INLINE` is similar (likely equivalent) to Kokkos's `KOKKOS_INLINE_FUNCTION`.

YAKL's `YAKL_LAMBDA` is similar to the Kokkos `KOKKOS_LAMBDA`, but there are important differences:
* In CUDA, YAKL specifies `#define YAKL_LAMBDA [=] __device__`, whereas Kokkos specifies `#define KOKKOS_LAMBDA [=] __host__ __device__`. YAKL differs here because of issues with calling atomics, which are only `__device__` functions in the hardware supported API. This will be a problem if trying to run `YAKL_LAMBDA` functions on the host, but that situation seems unlikely because explicitly creating a `YAKL_INLINE` function would seemingly suffice better in that case.

## Compiling with YAKL

### CMake

To use YAKL in another CMake project, the following is a template workflow to use in your CMakeLists.txt, assuming a target called `TARGET` and a list of C++ source files called `${CXX_SRC}`, and a yakl clone in the directory `${YAKL_HOME}`.

```cmake
# YAKL_ARCH can be CUDA, HIP, SYCL, OPENMP45, or empty (for serial CPU backend)
set(YAKL_ARCH "CUDA")  
# Set YAKL_[ARCH_NAME]_FLAGS where ARCH_NAME is CUDA, HIP, SYCL, OPENMP45, or CXX (for CPU targets)
set(YAKL_CUDA_FLAGS "-O3 -arch sm_70 -ccbin mpic++")
# Add the YAKL library and perform other needed target tasks
add_subdirectory(${YAKL_HOME} ./yakl)
# Set YAKL properties on the C++ source files in a target
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(TARGET)
message(STATUS "YAKL Compiler Flags: ${YAKL_COMPILER_FLAGS}")
```

YAKL's `yakl_process_target()` macro processes the target's C++ source files, and it will automtaically link the `yakl` library target into the `TARGET` you pass in. It also sets the CXX and CUDA (if `YAKL_ARCH == "CUDA"`) C++ standards to C++14 for the target, and it defines a `${YAKL_COMPILER_FLAGS}` variable you can query for the C++ flags applied to the target's C++ source files.

You can add `-DYAKL_DEBUG` to enable some YAKL compiler flags options, and this is valid on both the CPU and the GPU.

When setting flags for YAKL and targets' C++ files that use YAKL, you need to specify `YAKL_<LANG>_FLAGS` before processing the target, where `<LANG>` is currently: `CUDA`, `HIP`, `SYCL`, `OPENMP45`, `OPENMP`, or `CXX`. YAKL also has some internal Fortran and C files for `gator_mod.F90` and `gptl*.c`. You can specify flags for these files with `YAKL_C_FLAGS` and `YAKL_F90_FLAGS`.

**Important**:
* **All** of the processed target's C++ files are processed with YAKL's flags and other tasks
* **No other flags** are included in YAKL's C++ source files after they are processed. This is important for CUDA targets because the `nvcc` compiler cannot handle duplicate flags (for reasons I cannot understand). Therefore, to be clear, the **only** flags used for processed C++ source files come from `YAKL_<LANG>_FLAGS`. `CMAKE_CXX_FLAGS`, for instance, are not used for C++ files processed with YAKL's CMake macros.

If you'd rather deal with a list of C++ source files, you can use `yakl_process_cxx_source_files("${files_list}")`, where `${files_list}` is a list of C++ source files you want to process with YAKL flags and cmake attributes. Be sure not to forget the **quotations** around the variable, or the list will not properly make its way to the macro.

### Traditional Makefile

To compile in a traditional make file, you need to handle the backend-specific flags yourself. For different hardware backens, you need to specify `-DYAKL_ARCH_[BACKEND]` where `[BACKEND] is CUDA, HIP, SYCL, OpenMP45`. If you do not specify any `-DYAKL_ARCH_`, then it will compile for a serial CPU backend.

## YAKL Timers

YAKL includes performance timers based on the General Purpose Timing Library (GPTL) by Jim Rosinski (https://github.com/jmrosinski/GPTL). To use timers, simple place these calls around the code you want to time:

```C++
yakl::timer_start("mylabel")
...
yakl::timer_stop("mylabel")
```

YAKL handles the initialization and output internally, but you do have to pass `-DYAKL_PROFILE` as a command line argument. At the end of the run, the timer data is printed to `stdout`, and it is written to a file `yakl_timer_output.txt`. GPTL automatically keeps track of nested timer calls and gives easy to read output that gives the total walltime as well as the min and max walltime among the calls.

To make profiling easier, you can also pass `-DYAKL_AUTO_PROFILE`, and YAKL will wrap GPTL timer calls around every named `parallel_for` call in the code. Since calls without string labels wouldn't be very informative in a set of timers, those are not included. 

**Important:** All timer calls involve an `fence()` operation to ensure GPU kernels are accurately timed. This can add significant overhead to latency-sensitive (small-workload) applications. Also, if you do not specify either `-DYAKL_PROFILE` or `-DYAKL_AUTO_PROFILE`, then all timer calls will become no-ops.

## `YAKL_SCOPE`

In C++, lambdas only capture by value variables defined in **local** scope. This is a problem in two separate cases: `this->var` and `::var`. In each of these cases, since they are not in local scope, C++ lambdas access them by referencing them from the CPU. This will cause an invalid memory address error inside a device kernel. To alleviate this, please use `YAKL_SCOPE( var , this->var );`  or  `YAKL_SCOPE( var , ::var );` to place the variable into local scope so that C++ lambdas copy it by value, making it valid in device memory when used in a device kernel. This statement goes before the `parallel_for`. For instance:

```C++
class Chicken {
  int liver;
  
  void peck() {
    YAKL_SCOPE( liver , this->liver );
    parallel_for( 1 , YAKL_LABMDA (int dummy) {
      liver++;
    });
  }
};
```

What the `YAKL_SCOPE()` macro does is create a local reference to the variable, e.g., `auto &var = this->var;`

## Handling Class Methods Called from Kernels

If you have a class method prefixed with `YAKL_INLINE`, meaning you intend to potentially call it from a `parallel_for` kenel, you must delcare it as `static`. This is not a firm requirement in CUDA, but it is in HIP and likely SYCL as well. Also, it makes sense from a C++ perspective. `static` member functions belong to the class itself, not any particular object or instantiation of that class. Therefore, it does not use the `this->` pointer. This idea is moot if you plan on inlining the function; however, strictly speaking, you shouln't open yourself up to the possibility of calling a function via the `this->` pointer. Therefore the function should be static.

## A Note on Thread Safety

If you want to create and destroy YAKL `Array` objects or allocate and free `Gator` pool memory within a threaded region (CPU threads), you will need to ensure that you compile the C++ files that use YAKL with OpenMP enabled (e.g., `-fopenmp` for GNU or `-qsmp=omp` for IBM XL). Please note this is not the same things as using `-DYAKL_ARCH_OPENMP`, which changes the backend to OpenMP. This is just adding OpenMP flags to YAKL so that it can ensure thread safety in the pool allocator and shared reference counter pointer in the `Array` classes.

There are complicated reasons why YAKL's thread safety must rely on OpenMP rather than something better like `std::mutex`. Some versions of CUDA require the `Array` class destructors to be valid on the device, and `std::mutex` is not valid on the device. Since destructors have reference counter decrement operations that must be protected from thread races, OpenMP is the only option due to CUDA restrictions.

For instance, if you have a Fortran program that has a threaded region calling YAKL C++ code that's targetting the GPU using nvcc + GNU, then you need to ensure you add `-fopenmp --forward-unknown-to-host-compiler --forward-unknown-to-host-linker` to `YAKL_CUDA_FLAGS` and make sure that `-fopenmp` is in the linker flags as well.

## Future Work

Please see the github Issues for planned future work on YAKL.

## Software Dependencies
* For Nvidia GPUs, you'll need to clone [CUB](https://nvlabs.github.io/cub/)
* For AMD GPUs, you'll need to clone:
  * [hipCUB](https://github.com/ROCmSoftwarePlatform/hipCUB)
  * [rocPIRM](https://github.com/ROCmSoftwarePlatform/rocPRIM)

