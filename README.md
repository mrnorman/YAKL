# YAKL: Yet Another Kernel Launcher
## A Simple C++ Kernel Launcher for Performance Portability

YAKL is designed to be similar to Kokkos but significantly simplified to make it easier to add new hardware backends quickly. The YAKL kernel launcher, `parallel_for`, will work on any object that can be validly accessed in GPU memory. This includes objects that were allocated in GPU memory and objects that use a shallow copy with a data pointer in GPU memory (like the YAKL Array class or the Kokkos View class). The two classes, `Array`, `SArray`, and the `yakl` kernel launchers can all be used more or less independently.

Keep in mind this is still a work in progress.

Limitations & differences compared to other portability frameworks:
* Tightly nested loops are always "collapsed" into a single level of parallelism. Multiple levels of parallelism are not supported.
* Only one data layout is supported by the `Array` class: contiguous with the right-most index varying the fastest.
  * This makes it easier to interoperate with other libraries
* Data memory space is simplified to "host" (i.e., CPU) and "device" (i.e., GPU).
* Currently no "sub-array" capabilities, but it is coming soon.
* For arrays of compile-time-known size, you have to use a separate `SArray` (Static Array) class.
  * `SArray`s are meant for small arrays on the stack, while `Array`s are meant to be larger and on the heap.
* Unmanaged Arrays are not supported yet.

Benefits compared to other portability frameworks:
* It works on AMD, and adding new backends is simple.
* Movement between CPU and GPU is simple.
* The `parallel_for` syntax for multiple tightly nested loops is clean:
  * `yakl::parallel_for(dim1, dim2, YAKL_LAMBDA (int i1, int i2) {...} );`

## Code Sample

The following loop would be ported to general accelerators with YAKL as follows:

```C++
#include "Array.h"
#include "YAKL.h"
#include <iostream>
typedef float real;
typedef yakl::Array<real,yakl::memHost> realArr;
inline void applyTendencies(realArr &state2, real const c0, realArr const &state0,
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
inline void applyTendencies(realArr &state2, real const c0, realArr const &state0,
                                             real const c1, realArr const &state1,
                                             real const ct, realArr const &tend,
                                             Domain const &dom) {
  // for (int l=0; l<numState; l++) {
  //   for (int k=0; k<dom.nz; k++) {
  //     for (int j=0; j<dom.ny; j++) {
  //       for (int i=0; i<dom.nx; i++) {
  yakl::parallel_for( numState , dom.nz , dom.ny , dom.nx , YAKL_LAMBDA (int l, int k, int j, int i) {
    state2(l,hs+k,hs+j,hs+i) = c0 * state0(l,hs+k,hs+j,hs+i) +
                               c1 * state1(l,hs+k,hs+j,hs+i) +
                               ct * dom.dt * tend(l,k,j,i);
  }); 
}
yakl::ParallelSum<real,yakl::memDevice> psum( numState*dom.nx*dom.ny*dom.nz );
real tot = psum( state2.data() );
std::cout << state2.createHostCopy();
std::cout << tot << std::endl;
```

## Using YAKL

If you want to use the YAKL Array class, you'll need to `#include "Array.h"`, and if you want to use the YAKL launchers, you'll need to `#include YAKL.h`. Preface functions you want to run on the accelerator with `YAKL_INLINE`, and preface lambdas you're passing to YAKL launchers with `YAKL_LAMBDA` (which does a capture by value for CUDA and HIP backends for Nvidia and AMD hardware, respectively). The `parallel_for` launcher is used as follows:

```C++
// for (int i=0; i<nThreads; i++) {
yakl::parallel_for( int nThreads , FunctorType &f );

// for (int i1=0; i1<n1; i1++) {
//   for (int i2=0; i2<n2; i2++) {
//     for (int i3=0; i3<n3; i3++) {
yakl::parallel_for( int n1 , int n2 , int n3 , YAKL_LAMBDA (int i1 , int i2, int i3) {...} );
```

The `Array` class is set up to handle two different memories: Host and Device, and you can seen an example of how to use these above as well as in the [awflCloud](https://github.com/mrnorman/awflCloud) codebase. Also, it uses C-style index ordering with no padding between elements.

Be sure to use `yakl::init()` at the beginning of the program and `yakl::finalize()` at the end.

## Compiling with YAKL

You currently have three choices for a device backend: HIP, CUDA, and serial CPU. To use different hardware backends, add the following CPP defines in your code. You may only use one. 

| Hardware      | CPP Flag       | 
| --------------|----------------| 
| AMD GPU       |`-D__USE_HIP__` | 
| Nvidia GPU    |`-D__USE_CUDA__`| 
| CPU Serial    | no flag        | 

To turn on array bounds checking, add `-DARRAY_DEBUG` to your compiler flags.

In your compile line, you'll need to include the YAKL source directory in your `CXX_FLAGS` (e.g., `-I $YAKL_ROOT`). Also, you need to add `$YAKL_ROOT/YAKL.cpp` to your list of source files and its corresponding object file to your list of object files.

## Handling Two Memory Spaces

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

## Array Reductions

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

## Asynchronicity

All YAKL calls are asynchronously launched in the "default" CUDA or HIP stream when run on the device. Array `deep_copy_to` calls also do the same. With the exception of the reduction `operator()`, you'll need to call `yakl::fence()` if you want to wait on the device operation to complete.

## Future Work

Plans for the future include:
* Adding [OpenCL](https://www.khronos.org/opencl/) and [OpenMP](https://www.openmp.org/) backends
* Adding atomic functions for min, max, and sum
* Improving the documentation and testing of YAKL

## Software Dependencies
All of these are included as submodules in this repo:
* For Nvidia GPUs, you'll need to clone [CUB](https://nvlabs.github.io/cub/)
* For AMD GPUs, you'll need to clone:
  * [hipCUB](https://github.com/ROCmSoftwarePlatform/hipCUB)
  * [rocPIRM](https://github.com/ROCmSoftwarePlatform/rocPRIM)

