# YAKL: Yet Another Kernel Launcher
## A C++ Kernel Launcher for Performance Portability

**This is still a work in progress**

YAKL uses `BuddyAllocator` code from Mark Berrill at Oak Ridge National Laboratory.

* [Overview](#overview)
* [Code Sample](#code-sample)
* [Using YAKL](#using-yakl)
  * [`parallel_for`](#parallel_for)
  * [`Array`](#array)
  * [Moving Data Between Two Memory Spaces](#moving-data-between-two-memory-spaces)
  * [`SArray` and `YAKL_INLINE`](#sarray-and-yakl_inline)
  * [Managed Memory](#managed-memory)
  * [Pool Allocator](#pool-allocator)
  * [Synchronization](#synchronization)
  * [Atomics](#atomics)
  * [Reductions (Min, Max, and Sum)](#reductions-min-max-and-sum)
  * [Fortran - C++ Interoperability with YAKL: `gator_mod.F90`](#fortran---c-interoperability-with-yakl-gator_modf90)
  * [Interoperating with Kokkos](#interoperating-with-kokkos)
  * [`FArray` and `FSArray` (Fortran Behavior in C++)](#farray-and-fsarray-fortran-behavior-in-c)
* [Compiling with YAKL](#compiling-with-yakl)
* [Future Work](#future-work)
* [Software Dependencies](#software-dependencies)

## Overview

The YAKL API is similar to Kokkos in many ways. It is simplified in many ways to make adding multiple hardware backends easier. Yet, YAKL's increased simplicity allows it to do some things that some other frameworks cannot such as provide Fortran interoperability, lighter weight exposure of atomics, handling of host-to-device and device-to-host transfers, and easier interoperability with external libraries such as MPI and I/O libraries. 

YAKL currently has backends for CPUs, Nvidia GPUs, and AMD GPUs.

With less than 3K lines of code, YAKL provides the following:

* **Pool Allocator**: An optional pool allocator based on Mark Berrill's `BuddyAllocator` class for device data
  * Supports `malloc`, `cudaMalloc`, `cudaMallocManaged`, `hipMalloc`, and `hipMallocHost` allocators
  * CUDA Managed memory also calls `cudaMemPrefetchAsync` on the entire pool
  * If the pool allocator is not used, YAKL still maintains internal host and device allocators with the afforementioned options
  * Specify `-D__MANAGED__` to turn on `cudaMallocManaged` for Nvidia GPUs and `hipMallocHost` for AMD GPUs
* **Fortran Bindings**: Fortran bindings for the YAKL internal / pool device allocators
  * For `real(4)`, `real(8)`, `int(4)`, `int(8)`, and `logical` arrays up to seven dimensions
  * Using Fortran bindings for Managed memory makes porting from Fortran to C++ on the GPU significantly easier
  * You can call `gator_init(bytes)` from `gator_mod` to initialize YAKL's pool allocator or just `gator_init()` to initialize YAKL's allocators without the pool.
  * Call `gator_finalize()` to deallocate YAKL's runtime pool and other miscellaneous data
  * When you specify `-D__MANAGED__ -D__USE_CUDA__`, all `gator_allocate(...)` calls will not only use CUDA Managed Memory but will also map that data in OpenACC and OpenMP offload runtimes so that the data will be left alone in the offload runtimes and left for the CUDA runtime to manage instead.
    * This allows you to use OpenACC and OpenMP offload without any data statements and still get efficient runtime and correct results.
    * This is accomplished with `acc_map_data(...)` in OpenACC and `omp_target_associate_ptr(...)` in OpenMP 4.5+
    * With OpenACC, this happens automatically, but with OpenMP offload, you need to also specify `-D_OPEMP45`
* **Multi-Dimensional Arrays**: An `Array` class for allocated multi-dimensional array data
  * Only one memory layout is supported for simplicity: contiguous with the right-most index varying the fastest
    * This makes library interoperability more straightforward
  * Up to eight dimensions are supported
  * `Array`s support two memory spaces, `memHost` and `memDevice`, specified as a template parameter.
  * The `Array` API is similar to a simplified Kokkos `View` but greatly simplified, particularly in terms of template arguments 
  * `Array`s use shallow move and copy constructors (sharing the data pointer rather than copying the data) to allow them to work in C++ Lambdas via capture-by-value in a separate device memory address space.
  * `Array`s use YAKL's internal allocators, meaning they can be allocated with the YAKL pool allocator
  * `Array`s are by default "owned" (allocated, reference counted, and deallocated), but there are "non-owned" constructors that can wrap existing data
  * There are no sub-array capabilities as of yet
  * Supports array index debugging to throw an error when indices are out of bounds
* **Multi-Dimensional Arrays On The Stack**: An `SArray` class for static arrays to be placed on the stack
  * This makes it easy to create low-overhead local arrays in kernels
  * Supports up to four dimensions, the sizes of which are template parameters
  * Supports array index debugging to throw an error when indices are out of bounds
  * Because `SArray` objects are inherently on the stack, they have no templated memory space specifiers
* **Fortran-style Multi-dimensional Arrays**: `FArray` and `FSArray` classes for allocated and stack Fortran-like arrays
  * Left-most index varies the fastest just like in Fortran
  * Arbitrary lower bounds that default to one
  * `parallel_for` kernel launchers that take arbitrary loop bounds and strides like Fortran loops.
* **Kernel Launchers**: `parallel_for` launchers
  * Similar syntax as the Kokkos `parallel_for`
  * Only supports one level of parallelism for simplicity, so your loops need to be collapsed
  * Multiple tightly-nested loops are supported through the `yakl::unpackIndices(...)` utility function, which splits a single index into multiple indices for you.
  * Supports CUDA, CPU serial, and HIP backends at the moment
  * `parallel_for` launches on the device are by default asynchronous in the CUDA and HIP default streams
  * There is an automatic `fence()` option specified at compile time with `-D__AUTO_FENCE__` to insert fences after every `parallel_for` launch
* **Atomics**: Atomic instructions
  * `atomicAdd`, `atomicMin`, and `atomicMax` are supported for 4- and 8-byte integers and reals
  * Whenever possible, hardwhare atomics are used, and software atomics using `atomicCAS` and real-to-integer conversions in the CUDA and HIP backends
  * Note this is a significant divergence from Kokkos, which specifies atomic accesses for entire arrays all of the time. With YAKL, you only perform atomic accesses when you need them at the development expense of having to explicitly use `atomic[Add|Min|Max]` in the kernel code yourself.
* **Reductions**: Parallel reductions using the Nvidia `cub` and AMD `hipCUB` (wrapper to `rocPRIM`) libraries
  * Temporary data for a GPU device reduction is allocated with YAKL's internal device allocators and owned by a class for your convenience. It is automatically deallocated when the class falls out of scope.
  * The user can create a class object and then reuse it to avoid multiple allocations during runtime
  * If the pool allocator is turned on, allocations are fast either way
  * The `operator(T *data)` function defaults to copying the result of the reduction to a host scalar value
  * The user can also use the `deviceReduce(T *data)` function to store the result into a device scalar location
* **Synchronization**
  * The `yakl::fence()` operation forces the host code to wait for all device code to complete

## Code Sample

The following loop would be ported to general accelerators with YAKL as follows:

```C++
#include "Array.h"
#include "YAKL.h"
#include <iostream>
typedef float real;
typedef yakl::Array<real,yakl::memHost> realArr;
void applyTendencies(realArr &state2, real const c0, realArr const &state0,
                                      real const c1, realArr const &state1,
                                      real const ct, realArr const &tend,
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
#include "Array.h"
#include "YAKL.h"
#include <iostream>
typedef float real;
typedef yakl::Array<real,yakl::memDevice> realArr;
void applyTendencies(realArr &state2, real const c0, realArr const &state0,
                                      real const c1, realArr const &state1,
                                      real const ct, realArr const &tend,
                                      Domain const &dom) {
  // for (int l=0; l<numState; l++) {
  //   for (int k=0; k<dom.nz; k++) {
  //     for (int j=0; j<dom.ny; j++) {
  //       for (int i=0; i<dom.nx; i++) {
  yakl::parallel_for( numState*dom.nz*dom.ny*dom.nx , YAKL_LAMBDA (int iGlob) {
    int l, k, j, i;
    yakl:unpackIndices( iGlob , numState,dom.nz,dom.ny,dom.nx , l,k,j,i );
    
    state2(l,hs+k,hs+j,hs+i) = c0 * state0(l,hs+k,hs+j,hs+i) +
                               c1 * state1(l,hs+k,hs+j,hs+i) +
                               ct * dom.dt * tend(l,k,j,i);
  }); 
  yakl::ParallelSum<real,yakl::memDevice> psum( numState*dom.nx*dom.ny*dom.nz );
  real tot = psum( state2.data() );
}
std::cout << state2.createHostCopy();
std::cout << tot << std::endl;
```

## Using YAKL

If you want to use the YAKL `Array` or `SArray` classes, you'll need to `#include "Array.h"` or `#include "SArray.h"`. If you want to use the YAKL launchers, atomics, allocators, or reductions you'll need to `#include YAKL.h`.

Be sure to use `yakl::init(...)` at the beginning of the program and `yakl::finalize()` at the end. If you do not call these functions, you will get errors during runtime for all `Array`s, `SArray`s, and device reductions. You may get errors for CUDA `parallel_for`s.

### `parallel_for`

Preface functions you want to run on the accelerator with `YAKL_INLINE`, and preface lambdas you're passing to YAKL launchers with `YAKL_LAMBDA` (which does a capture by value for CUDA and HIP backends for Nvidia and AMD hardware, respectively). The recommended use case for the `parallel_for` launcher is:

```C++
#include "YAKL.h"
...
// for (int j=0; j<ny; j++) {
//   for (int i=0; i<nx; i++) {
yakl::parallel_for( ny*nx , YAKL_LAMBDA (int iGlob) {
  int j, i;
  yakl::unpackIndices( iGlob , ny,nx , j,i );
  
  // Loop body goes here
});

// for (int i=0; i<nx; i++) {
yakl::parallel_for( nx , YAKL_LAMBDA (int i) {
  // Loop body goes here
});
```

The `parallel_for` launcher is useful for every type of loop except for prefix sums and reductions. You can embed atomics in `parallel_for` as you'll see a few sections down.

### `Array`

The `Array` class is set up to handle two different memories: Host and Device, and you can seen an example of how to use these above as well as in the [miniWeather](https://github.com/mrnorman/miniWeather) codebase.

The `Array` class can be owned or non-owned. The constructors are:

```C++
# Owned (Allocated, reference counted, and deallocated by YAKL)
yakl::Array<T type,int memSpace>(char const *label, int dim1, [int dim2, ...]);

# Non-Owned (NOT allocated, reference counted, or deallocated)
# Use this to wrap existing contiguous data pointers (e.g., from Fortran)
yakl::Array<T type,int memSpace>(char const *label, T *ptr, int dim1, [int dim2, ...]);
```

Data is accessed in an `Array` object via the `(ind1[,ind2,...])` operator with the right-most index varying the fastest.

YAKL `Array` objects use shallow copies in the move and copy constructors. To copy the data from one `Array` to another, use the `Array::deep_copy_to(Array &destination)` member function. This will copy data between different memory spaces (e.g., `cudaMemcpy(...)`) for you depending on the memory spaces of the `Array` objects.

To create a host copy of an `Array`, use `Array::createHostCopy()`, and to create a device copy, use `Array::createDeviceCopy()`.

### Moving Data Between Two Memory Spaces

The intent of YAKL is to mirror copies of the `Array` class between two distinct memory spaces: Host (i.e., main memory) and Device (e.g., GPU memory). There are currently four member functions of the `Array` class to help with data movement:

```C++
// Create a copy of this Array class in Host Memory, and pass that copy back as a return value.
template<class T> Array<T,yakl::memHost> createHostCopy();

// Create a copy of this Array class in Device Memory, and pass that copy back as a return value.
template<class T> Array<T,yakl::memDevice> createDeviceCopy();

// Copy the data from this Array pointer to the Host Array's pointer (Host Array must already exist)
template<class T> void deep_copy_to(Array<T,memHost> lhs);

// Copy the data from this Array pointer to the Device Array's pointer (Device Array must already exist)
template<class T> void deep_copy_to(Array<T,memDevice> lhs);
```

### `SArray` (and `YAKL_INLINE`)

To declare local data on the stack inside a kernel, you'll use the `SArray` class. For instance:

```C++
typedef float real;
yakl::Array<real,memDevice> state("state",nx+2*hs);
yakl::Array<real,memDevice> coefs("coefs",nx,ord);
...
YAKL_INLINE void recon( SArray<real,ord> const &stencil , 
                        SArray<real,ord> &coefs ) { ... }
yakl::parallel_for( nx , YAKL_LAMBDA (int i) {
  yakl::SArray<real,ord> stencil, coefs;
  for (int ii=0; ii<ord; ii++) { stencil(ii) = state(i+ii); }
  recon(stencil,coefs);
  for (int ii=0; ii<ord; ii++) { coefs(i,ii) = coefs(ii); }
});
```

Data is accessed in an `SArray` object via the `(ind1[,ind2,...])` operator with the right-most index varying the fastest.

Note that `SArray` objects are inherently placed on the stack of whatever context they are declared in. Because of this, it doesn't make sense to allow a templated memory specifier (i.e., `yakl::memHost` or `yakl::memDevice`). If used in a CUDA kernel, `SArray` objects will be placed onto the kernel stack frame. if used in CPU functions, they will be placed in the CPU function's stack.

### Managed Memory

To use CUDA Managed Memory or HIP pinned memory (which performs terribly but does make CPU data available to the GPU), add `-D__MANAGED__` to the compiler flags. This will make your life much easier when porting a Fortran code to a GPU-enabled C++ code because all data allocated with the Fortran hooks of YAKL will be avilable on the CPU and the GPU without having to transfer it explicitly.

### Pool Allocator

You can initialize a pool allocator by initializing YAKL with a parameter for the size of the pool in bytes.

**Beware**: The pool size must be large enough because YAKL does not support multiples pools or resizing the existing pool.

```C++
  typedef float real;
  yakl::init( nz*ny*nx*100*sizeof(real) );
```

To disable the pool allocator, simply call `yakl::init()` without any parameters.

### Synchronization

Currently YAKL only supports asynchronicity through the default CUDA stream. To synchronize the host code with code running on a GPU device, use the `yakl::fence()` functions which no-ops in host code and calls `cudaDeviceSynchronize()` on Nvidia GPUs and `hipDeviceSynchronize()` on AMD GPUs.

### Atomics

When you need atomic operations in YAKL, you can use the `atomicAdd`, `atomicMin`, and `atomicMax` routines. The most common use-case for atomics is some form of partial reduction over some of the indices of an array but not all. For example:

```C++
yakl::Array<real,memDevice> :: arrLarge("arrLarge",nz,ny,nx,ncrm);
yakl::Array<real,memDevice> :: arrSmall("arrLarge",nz,ncrm);

...

// for (int k=0; k<nzm; k++) {
//   for (int j=0; j<ny; j++) {
//     for (int i=0; i<nx; i++) {
//       for (int icrm=0; icrm<ncrms; icrm++) {
yakl::parallel_for( nzm*ny*nx*ncrms , YAKL_LAMBDA (int iGlob) {
  int k, j, i, icrm;
  yakl::unpackIndices( iGlob , nzm,ny,nx,ncrms , k,j,i,icrm );

  // The operation below is a race condition in parallel. It needs an atomicMax
  // arrSmall(k,icrm) = max( arrSmall(k,icrm) , arrLarge(k,j,i,icrm) );
  
  yakl::atomicMax( arrSmall(k,icrm) , arrLarge(k,j,i,icrm) );
});
```

As a rule, if you ever see anything on the left-hand-side of an `=` with **fewer indices than you have surrounding loops**, then you're going to have a race condition that requires an atomic access.

### Reductions (Min, Max, and Sum)

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

### Fortran - C++ interoperability with YAKL: `gator_mod.F90`

We provide the `gator_mod` Fortran module to interface with YAKL's internal device allocator. To use it, you'll first need to make all "automatic" fortran arrays into `allocatable` arrays and explicitly allocat them:

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

Next, to interface with `gator_mod`, you'll need to transform all `allocatables` into `pointer, contiguous`. The `contiguous` attribute is recommended because some compilers will perform more optimizations on Fortran pointers when it is present. The transformation will become:

```Fortran
use gator_mod, only: gator_allocate, gator_deallocate

real, pointer, contiguous :: var1(:,:,:)
real, pointer, contiguous :: var2(:,:,:)
...
call gator_allocate( var1 , (/ ubnd_x-lbnd_x+1 , ubnd_y-lbnd_y+1 , ubnd_z-lbnd_z+1 /) , (/-1,-1,-1/) )
call gator_allocate( var2 , (/nx,ny,nz/) )
...
call gator_deallocate(var1)
call gator_deallocate(var2)
```

Note that YAKL **does** support non-1 lower bounds in Fortran in `gator_allocate()`. The first set of indices specify the total extents of each dimension. The second set of indices specify the lower bounds of each dimension in case they aren't the default 1 lower bounds of Fortran. 

If a Fortran routine uses module data, when porting to C++, it is often easiest to mirror the Fortran practice. In this case, it is best to pass to allocate the Fortran data with `gator_allocate` and then immediately pass it to a C++ wrapping function, which will wrap it in unmanaged YAKL `Array` classes. With the above data, you will have the following code to do that:

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
  integer(c_int), bind(C) :: ubnd_x, ubnd_y, ubnd_z, lbnd_x, lbnd_y, lbnd_z, nx, ny, nz
end module cpp_interface_mod
```

Fortran assumed size arrays (i.e., `dimension(*)`) are very convenient because an array of any dimension can be legally passed to the subroutine. Since Fortran passes by reference by default, the location of the data is passed to this array.

All variables in Fortran modules need to be moved to `iso_c_binding` types such as: `real(c_float)`, `real(c_double)`, `integer(c_int)`, and `logical(c_bool)`. All module-level scalars can directly be bound to C with `bind(C)`. Arrays, however, need to be passed to a wrapping routine.

In C++, you would have a header file, `fortran_data.h` with the following:

```C++
#ifndef __FORTRAN_DATA_H__
#define __FORTRAN_DATA_H__
typedef yakl::Array<real,yakl::memDevice> umgReal3d;
extern "C" void wrap_arrays(float *var1, float *var2);
extern real3d var1;
extern real3d var2;
extern int ubnd_x, ubnd_y, ubnd_z, lbnd_x, lbnd_y, lbnd_z, nx, ny, nz;
extern int dim_x, dim_y, dim_z;
extern int offset_x, offset_y, offset_z;
#endif
```

And you would have a source file, `fortran_data.cpp`, with:

```C++
#include "fortran_data.h"
int dimx = ubnd_x - lbnd_x + 1;
int dimy = ubnd_y - lbnd_y + 1;
int dimz = ubnd_z - lbnd_z + 1;
int offset_x = 1-lbnd_x;
int offset_y = 1-lbnd_y;
int offset_z = 1-lbnd_z;

extern "C" void wrap_arrays(float *var1_p, float *var2_p) {
  // These create un-owned YAKL Arrays using existing allocations from Fortran
  var1 = real3d( "var1" , var1_p , dimz , dimy , dimx );
  var2 = real3d( "var2" , var2_p , nz   , ny   , nx   );
}
```

Notice that the order of the dimensions is reversed in the C++ code. This is because the fastest varying index is now the right-most, not the left-most. Also notice that the `[l|u]bnd_[x|y|z]` variables are declared as `extern` in the header file but aren't mentioned in the source file. This is because with the `bind(C)` attribute in Fortran, they already exist in the object files under those names. Also, the `wrap_arrays` function in C++ must be declared as `extern "C"` to force the compiler to give it a name in the object file that is expected by Fortran with the `iso_c_binding`. 

Finally, the `var2` array will be indexed from indices `0` to `nx-1`, `0` to `ny-1`, and `0` to `nz-1` in the C++ code in the x, y, and z dimensions, respectively. But since `var1` has non-standard Fortran lower bounds, you must add the `offset_x`, `offset_y`, and `offset_z` integers to the indexing in the x, y, and z dimensions, respectively, when indexing using zero-based loops like `var1`.

When Fortran data is passed by parameter, you can use the un-owned `Array` constructors inside the C++ functions following the same ideas regarding non-standard Fortran lower bounds.

### Interoperating with Kokkos

YAKL can interoperate with Kokkos in a number of ways. The YAKL `Array` is equivalent to the Kokkos `View` in the following mappings:

* `Array<T,memHost>` with neither `-D__USE_CUDA__` nor `-D__USE_HIP__`
  * This can be compatible with any dimension of Kokkos `View` in `HostSpace` with `LayoutRight` such as `View<T*,LayoutRight,HostSpace>` or `View<real****,LayoutRight,HostSpace>`
  * YAKL does not need to template on the number of dimensions, which simplifies things. But it complicates compatibility with Kokkos `View` objects somewhat.
  * A good practice is to try to `typedef` YAKL `Array` objects to `real1d`, `real2d`, etc. and to try to keep those correct to the actual data being used. If you do this, you can easily map it to a Kokkos `View` of the correct dimension later, and you will get compile-time or run-time errors if you didn't do the dimensionality correctly.
* `Array<T,memDevice>` with `-D__USE_CUDA__`
  * This is compatible with any dimension of Kokkos `View<T*,LayoutRight,CudaSpace>`
* `Array<T,memDevice>` with `-D__USE_CUDA__ -D__MANAGED__`
  * This is compatible with any dimension of Kokkos `View<T*,LayoutRight,CudaUVMSpace>`

Both Kokkos and YAKL have unmanaged / un-owned multi-dimensional arrays, so you can wrap equivalent types using the data pointer, which each expose via `Array::data()` and `View::data()`

YAKL `parallel_for` launchers can use Kokkos `Views` without issue, and Kokkos `parallel_for` and `parallel_reduce` launchers can use YAKL `Array` objects without issue.

You can use Kokkos View data in YAKL's reductions via the `View::data()` pointer so long as the Kokkos `View` is congiguous, which you can determine via the `View::span_is_contiguous()` member function. It's prefereable for performance that the Kokkos `View` also have the `LayoutRight` attribute as well.

You can use YAKL atomic functions inside Kokkos `parallel_for` launchers. 

You can use YAKL `SArray` objects inside Kokkos `parallel_for` launchers.

YAKL and Kokkos `fence()` operations are pretty much equivalent and can be used in either framework.

YAKL's `YAKL_INLINE` is equivalent to Kokkos's `KOKKOS_INLINE_FUNCTION`.

YAKL's `YAKL_LAMBDA` is similar to the Kokkos `KOKKOS_LAMBDA`, but there are important differences:
* In CUDA, YAKL specifies `#define YAKL_LAMBDA [=] __device__`, whereas Kokkos specifies `#define KOKKOS_LAMBDA [=] __host__ __device__`. YAKL differs here because of issues with calling atomics, which are only `__device__` functions in the hardware supported API.
* On the host, YAKL specifies `#define YAKL_LAMBDA [&]`, wherease Kokkos specifies `#define KOKKOS_LAMBDA [=]`. This means that on the CPU, YAKL lambdas can capture by reference, but Kokkos lambdas cannot.

YAKL's build system in CMake simply takes the C++ files and tells CMake to compile them as if they were CUDA. You don't need to change the C++ compiler.

The Kokkos build system is more invasive and complex, requiring you to change the `CMAKE_CXX_COMPILER` for all files in the project to use the `nvcc_wrapper` shipped with the Kokkos repo, and compiling different source files depending upon CMake options you pass before adding the library. Theoretically, a CMake system using the Kokkos build requirements should handle the YAKL source files, but you probably can't use YAKL's normal CMake library. Integrating YAKL into a larger project should be very simple because YAKL's C++ interface only has two source files: `BuddyAllocator.cpp` and `YAKL.cpp`, and its Fortran interface only has one: `gator_mod.F90`.

## `FArray`, and `FSArray`, and Fortran-like `parallel_for` (Fortran Behavior in C++)

YAKL also has the `FArray` and `FSArray` classes to make porting Fortran code to C++ simpler. It's time-consuming and error-prone to have to reorder the indices of every array to row-major and change all indexing to match C's always-zero lower bound. Also, Fortran allows array slicing and non-one lower bounds, which can make porting very tedious. Look-up tables and intermediate index arrays make this even harder.

The `FArray` and `FSArray` classes allow any arbritrary lower bound, defaulting to one, and the left-most index always varies the fastest like in Fortran. You can create them as follows:

**`FArray`**:
* `FArray<float,yakl::memSpace> arr( "label" , int dim1 [, int dim2 , ...] )
  * Equivalent to Fortran: `real, allocatable :: arr(:[,:,...]); allocate(arr(dim1[,dim2,...])`
  * Creates an owned array
* `FArray<float,yakl::memSpace> arr( "label" , float *data_p , int dim1 [, int dim2 , ...] )
  * Same as before but non-owned (wraps an existing contiguous data pointer)
* `FArray<double,yakl::memSpace> arr( "label" , {-1,52} [ , {0,43} , ...] )
  * Equivalent to Fortran: `real(8), allocatable :: arr(:[,:,...]); allocate(arr(-1:52 [ , 0:43 , ...])`
  * Lower and upper bounds are specified as integer pairs and accepted as `std::vector<int>` dummy variables
* `FArray<double,yakl::memSpace> arr( "label" , double *data_p , {-1,52} [ , {0,43} , ...] )
  * Same as before but non-owned (wraps an existing contiguous data pointer)
* `FArray::slice(COLON,COLON,ind1,ind2)`
  * Equivalent to Fortran array slicing: `arr(:,:,ind1,ind2)`
  * Only works on simple, *contiguous* array slices with *entire dimensions* (not partial dimensions) sliced out
  * E.g., `arr(0:5,4,7)`, though contiguous is not supported.
  * If you want to **write** to the array slice passed to a function, you must save it as a temporary variable first and pass the temporary variable: E.g., `auto tmp = arr.slice(COLON,COLON,ind1,ind2);  myfunc(tmp);`
  * If you're reading from the array slice, you can pass it directly inline.

**`FSArray`**:
* Must template on the lower and upper bounds to place it on the stack.
* `FSArray<float,bnd<-hs,hs>> stencil;`
  * Equivalent to the Fortran "automatic" array: `real :: stencil(-hs:hs)`
  * Low-overhead, placed ont he stack
  * You **must** specify two indices for each of up to four dimension in the template: the lower and upper bounds for that dimension

**`parallel_for` (Fortran style)**:
```C++
yakl::parallel_for( Bounds({-1,30,3},{0,29}) , YAKL_LAMBDA ( int indices[] ) {
  int j, i;
  yakl::storeIndices( indices , j,i );
  
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

## Compiling with YAKL

### CMake

To use YAKL in another CMake project, insert the following into your CMakeLists.txt

```cmake
add_subdirectory(/path/to/yakl  path/to/build/directory)
include_directories(/path/to/yakl)
```

The following variables can be defined before the `add_subdirectory()` command: `${ARCH}`, `${CUDA_FLAGS}`, `${HIP_FLAGS}`

To compile YAKL for a device target, you'll need to specify `-DARCH="CUDA"` or `-DARCH="HIP"` for Nvidia and AMD GPUs, respectively. If you don't specify either of these, then YAKL will compile for the CPU in serial.

You can further specify `-DCUDA_FLAGS="-D__MANAGED__ -arch sm_70"` or other flags if you wish. if you omit these flags, YAKL will compile without Managed Memory and without a specific GPU target. If you use double precision `atomicAdd`, and you're using a modern Nvidia GPU (`sm_60` or greater), you will want to specify `-arch` to ensure good performance. Again, specify these before running the CMake `add_subdirectory` command.

You can specify `-DHIP_FLAGS="..."` to pass in more options to AMD's `hipcc` compiler.

If you specify `-DARCH="CUDA"`, then you **must** specify `-DYAKL_CUB_HOME=/path/to/cub`.

If you specify `-DARCH="HIP"`, then you **must** specify `-DYAKL_HIPCUB_HOME=/path/to/hipCUB -DYAKL_ROCPRIM_HOME=/path/to/rocPRIM`

After running the `add_subdirectory()` command, YAKL will export four variables into parent scope for you to use: `${YAKL_CXX_SOURCE}`, `${YAKL_F90_SOURCE}`, `${YAKL_SOURCE}`, and `${YAKL_CXX_FLAGS}`. The C++ flags are given to you since it's almost certain you'll need to compile C++ source files that link in YAKL headers, and these will needs those flags.

You can compile C++ files that use YAKL header files with the following:

```cmake
set_source_files_properties(whatever.cpp PROPERTIES COMPILE_FLAGS "${YAKL_CXX_FLAGS}")
if ("${ARCH}" STREQUALS "CUDA")
  set_source_files_properties(whatever.cpp PROPERTIES LANGUAGE CUDA)
endif()
```

Also, the YAKL source files are given to you in case you want to compile YAKL is a different way yourself, for instance, if the YAKL `CMakeLists.txt` isn't playing nicely with your own CMake build system for some reason.

### Traditional Makefile

You currently have three choices for a device backend: HIP, CUDA, and serial CPU. To use different hardware backends, add the following CPP defines in your code. You may only use one, no mixing of the backends. 

| Hardware      | CPP Flag       | 
| --------------|----------------| 
| AMD GPU       |`-D__USE_HIP__` | 
| Nvidia GPU    |`-D__USE_CUDA__`| 
| CPU Serial    | neither of the above two | 

Passing `-DARRAY_DEBUG` will turn on array index debugging for `Array` and `SArray` objections. **Beware** that this only works on host code at the moment, so do not pass `-DARRAY_DEBUG` at the same time as passing `-D__USE_CUDA__` or `-D__USE_HIP__`.

Pasing `-D__MANAGED__` will trigger `cudaMallocManaged()` in tandem with `-D__USE_CUDA__` and `hipMallocHost()` in tandem with `-D__USE_HIP__`.

Passing `-D__AUTO_FENCE__` will automatically insert a `yakl::fence()` after every `yakl::parallel_for` launch.

You will have to explicitly pass `-I/path/to/cub` if you specify `-D__USE_CUDA__`  and  `-I/path/to/hipCUB -I/path/to/rocPRIM` if you specify `-D__USE_HIP__`.

## Future Work

Plans for the future include:
* Add [OpenCL](https://www.khronos.org/opencl/) and [OpenMP](https://www.openmp.org/) backends
* Improve the documentation and testing of YAKL
* Improve YAKL's CMake build system to be more generic

## Software Dependencies
* For Nvidia GPUs, you'll need to clone [CUB](https://nvlabs.github.io/cub/)
* For AMD GPUs, you'll need to clone:
  * [hipCUB](https://github.com/ROCmSoftwarePlatform/hipCUB)
  * [rocPIRM](https://github.com/ROCmSoftwarePlatform/rocPRIM)

