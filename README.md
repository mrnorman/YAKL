# YAKL: Yet Another Kernel Launcher
## A Minimally Invasive C++ Performance Portability Library

YAKL is designed to be similar to Kokkos but significantly simplified to make it easier to add new hardware backends quickly. The YAKL kernel launcher, `parallel_for`, will work on any object that can be validly accessed in GPU memory. This includes objects that were allocated in GPU memory and objects that use a shallow copy with a data pointer in GPU memory (like the YAKL Array class or the Kokkos View class).

## Simple Code Sample

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
  for (int l=0; l<numState; l++) {
    for (int k=0; k<dom.nz; k++) {
      for (int j=0; j<dom.ny; j++) {
        for (int i=0; i<dom.nx; i++) {
          state2(l,hs+k,hs+j,hs+i) = c0 * state0(l,hs+k,hs+j,hs+i) +
                                     c1 * state1(l,hs+k,hs+j,hs+i) +
                                     ct * dom.dt * tend(l,k,j,i);
        }
      }
    }
  }
}
std::cout << state2;
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
  yakl::parallel_for( numState*dom.nz*dom.ny*dom.nx , YAKL_LAMBDA (int iGlob) {
    int l, k, j, i;
    yakl::unpackIndices(iGlob,numState,dom.nz,dom.ny,dom.nx,l,k,j,i);
    state2(l,hs+k,hs+j,hs+i) = c0 * state0(l,hs+k,hs+j,hs+i) +
                               c1 * state1(l,hs+k,hs+j,hs+i) +
                               ct * dom.dt * tend(l,k,j,i);
  }); 
}
std::cout << state2.createHostCopy();
```

## Using YAKL

If you want to use the YAKL Array class, you'll need to `#include "Array.h"`, and if you want to use the YAKL launchers, you'll need to `#include YAKL.h`. Preface functions you want to run on the accelerator with `YAKL_INLINE`, and preface lambdas you're passing to YAKL launchers with `YAKL_LAMBDA` (which does a capture by value for CUDA and HIP backends for Nvidia and AMD hardware, respectively). The `parallel_for` launcher is used as follows:

```C++
yakl::parallel_for( int nThreads , FunctorType &f );
```

And the index unpacking utility for tightly nested loops is:

```C++
yakl::unpackIndices(int globalIndex, int dimSize1, [int dimSize2, ...], int index1, [int index2, ...])
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

## Handling Two Memory Spaces

The intent of YAKL is to mirror copies of the `Array` class between two distinct memory spaces: Host (i.e., main memory) and Device (e.g., GPU memory). There are currently four member functions of the `Array` class to help with data movement:

```C++
// Create a copy of this Array class in Host Memory, and pass that copy back as a return value.
template<class T> Array<T,yakl::memHost> createHostCopy();

// Create a copy of this Array class in Device Memory, and pass that copy back as a return value.
template<class T> Array<T,yakl::memDevice> createDeviceCopy();

// Copy the data from this Array pointer to the Host Array's pointer (Host Array must already exist)
template<class T> void deep_copy(Array<T,memHost> lhs);

// Copy the data from this Array pointer to the Device Array's pointer (Device Array must already exist)
template<class T> void deep_copy(Array<T,memDevice> lhs);
```

## Array Reductions

YAKL provides efficient min, max, and sum array reductions using [CUB](https://nvlabs.github.io/cub/) and [hipCUB](https://github.com/ROCmSoftwarePlatform/hipCUB) for Nvidia and AMD GPUs. Because these implementations require temporary storage, a design choice was made to expose reductions through class objects. Upon construction, you must specify the size (number of elements to reduce), type (`template <class T>`) of the array that will be reduced, and the memory space (via template parameter, `yakl::memHost` or `yakl::memDevice`) of the array to be reduced. The constructor then allocates memory for the temporary storage. Then, you run the reduction on an array of that size using `T operator()(T *data)`, which returns the result of the reduction in host memory. When the object goes out of scope, it deallocates the data for you. The array reduction objects are not sharable and implements no shallow copy. An example reduction is below:

```C++
Array<real> dt3d;
// Fill dt3d
yakl::ParallelMin<real,yakl::memDevice> pmin( nx*ny*nz );
dt = pmin( dt3d.data() );
```

If you want to avoid copying the result back to the host, you can run the `void deviceReduce(T *data, T *rslt)` member function, where the `rslt` pointer is allocated in device memory. An example is below:

```C++
Array<real> dt3d;
T *dtDev;
// Allocate dtDev on device
// Fill dt3d
yakl::ParallelMin<real,yakl::memDevice> pmin( nx*ny*nz );
pmin.deviceReduce( dt3d.data() , dtDev );
```

## Future Work

Plans for the future include:
* Adding [OpenCL](https://www.khronos.org/opencl/) and [OpenMP](https://www.openmp.org/) backends
* Adding atomic functions for min, max, and sum
* Improving the documentation of YAKL

