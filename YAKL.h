
#ifndef __YAKL_H_
#define __YAKL_H_

#include <iostream>
#include <algorithm>

#ifdef __NVCC__
  #define _YAKL __host__ __device__
#else
  #define _YAKL
#endif


namespace yakl {

  typedef unsigned long ulong;
  typedef unsigned int  uint;


  // Unpack 2D indices
  _YAKL void unpackIndices(uint iGlob, uint n1, uint n2, uint &i1, uint &i2) {
    i1 = (iGlob/(n2))     ;
    i2 = (iGlob     ) % n2;
  }


  // Unpack 3D indices
  _YAKL void unpackIndices(uint iGlob, uint n1, uint n2, uint n3, uint &i1, uint &i2, uint &i3) {
    i1 = (iGlob/(n3*n2))     ;
    i2 = (iGlob/(n3   )) % n2;
    i3 = (iGlob        ) % n3;
  }

  
  // Unpack 4D indices
  _YAKL void unpackIndices(uint iGlob, uint n1, uint n2, uint n3, uint n4, uint &i1, uint &i2, uint &i3, uint &i4) {
    i1 = (iGlob/(n4*n3*n2))     ;
    i2 = (iGlob/(n4*n3   )) % n2;
    i3 = (iGlob/(n4      )) % n3;
    i4 = (iGlob           ) % n4;
  }

  
  // Unpack 5D indices
  _YAKL void unpackIndices(uint iGlob, uint n1, uint n2, uint n3, uint n4, uint n5, uint &i1, uint &i2, uint &i3, uint &i4, uint &i5) {
    i1 = (iGlob/(n5*n4*n3*n2))     ;
    i2 = (iGlob/(n5*n4*n3   )) % n2;
    i3 = (iGlob/(n5*n4      )) % n3;
    i4 = (iGlob/(n5         )) % n4;
    i5 = (iGlob              ) % n5;
  }

  
  // Unpack 6D indices
  _YAKL void unpackIndices(uint iGlob, uint n1, uint n2, uint n3, uint n4, uint n5, uint n6, uint &i1, uint &i2, uint &i3, uint &i4, uint &i5, uint &i6) {
    i1 = (iGlob/(n6*n5*n4*n3*n2))     ;
    i2 = (iGlob/(n6*n5*n4*n3   )) % n2;
    i3 = (iGlob/(n6*n5*n4      )) % n3;
    i4 = (iGlob/(n6*n5         )) % n4;
    i5 = (iGlob/(n6            )) % n5;
    i6 = (iGlob                 ) % n6;
  }

  
  // Unpack 7D indices
  _YAKL void unpackIndices(uint iGlob, uint n1, uint n2, uint n3, uint n4, uint n5, uint n6, uint n7, uint &i1, uint &i2, uint &i3, uint &i4, uint &i5, uint &i6, uint &i7) {
    i1 = (iGlob/(n7*n6*n5*n4*n3*n2))     ;
    i2 = (iGlob/(n7*n6*n5*n4*n3   )) % n2;
    i3 = (iGlob/(n7*n6*n5*n4      )) % n3;
    i4 = (iGlob/(n7*n6*n5         )) % n4;
    i5 = (iGlob/(n7*n6            )) % n5;
    i6 = (iGlob/(n7               )) % n6;
    i7 = (iGlob                    ) % n7;
  }

  
  // Unpack 8D indices
  _YAKL void unpackIndices(uint iGlob, uint n1, uint n2, uint n3, uint n4, uint n5, uint n6, uint n7, uint n8, uint &i1, uint &i2, uint &i3, uint &i4, uint &i5, uint &i6, uint &i7, uint &i8) {
    i1 = (iGlob/(n8*n7*n6*n5*n4*n3*n2))     ;
    i2 = (iGlob/(n8*n7*n6*n5*n4*n3   )) % n2;
    i3 = (iGlob/(n8*n7*n6*n5*n4      )) % n3;
    i4 = (iGlob/(n8*n7*n6*n5         )) % n4;
    i5 = (iGlob/(n8*n7*n6            )) % n5;
    i6 = (iGlob/(n8*n7               )) % n6;
    i7 = (iGlob/(n8                  )) % n7;
    i8 = (iGlob                       ) % n8;
  }


  // These unfotunately cannot live inside classes in CUDA :-/
  #ifdef __NVCC__
  template <class F, class... Args> __global__ void parallelForCUDA(ulong const nIter, F &f, Args&&... args) {
    ulong i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < nIter) {
      f( i , args... );
    }
  }
  #endif


  uint const targetCUDA      = 1;
  uint const targetCPUSerial = 2;


  template <class F, class... Args> inline void parallelForCPU(ulong const nIter, F &f, Args&&... args) {
    for (ulong i=0; i<nIter; i++) {
      f( i , args... );
    }
  }


  template <class F, class... Args> inline void parallelFor(ulong const nIter, F &f, Args&&... args) {
    if        (target == targetCUDA) {
      #ifdef __NVCC__
        parallelForCUDA <<< (uint) nIter/vectorSize+1 , vectorSize , 0 , myStream >>> ( nIter , f , args... );
      #else
        std::cerr << "ERROR: "<< __FILE__ << ":" << __LINE__ << "\n";
        std::cerr << "Attempting to launch CUDA without nvcc\n";
      #endif
    } else if (target == targetCPUSerial ) {
      parallelForCPU( nIter , f , args... );
    }
  }

  template <class F, class... Args> inline void parallelFor(ulong const nIter, F const &f, Args&&... args) {
    if        (target == targetCUDA) {
      #ifdef __NVCC__
        parallelForCUDA <<< (uint) nIter/vectorSize+1 , vectorSize , 0 , myStream >>> ( nIter , f , args... );
      #else
        std::cerr << "ERROR: "<< __FILE__ << ":" << __LINE__ << "\n";
        std::cerr << "Attempting to launch CUDA without nvcc\n";
      #endif
    } else if (target == targetCPUSerial ) {
      parallelForCPU( nIter , f , args... );
    }
  }


  #ifdef __NVCC__
    __device__ __forceinline__ void atomicMin(float *address , float value) {
      int oldval, newval, readback;
      oldval = __float_as_int(*address);
      newval = __float_as_int( __int_as_float(oldval) < value ? __int_as_float(oldval) : value );
      while ( ( readback = atomicCAS( (int *) address , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __float_as_int( __int_as_float(oldval) < value ? __int_as_float(oldval) : value );
      }
    }

    __device__ __forceinline__ void atomicMin(double *address , double value) {
      unsigned long long oldval, newval, readback;
      oldval = __double_as_longlong(*address);
      newval = __double_as_longlong( __longlong_as_double(oldval) < value ? __longlong_as_double(oldval) : value );
      while ( ( readback = atomicCAS( (unsigned long long *) address , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __double_as_longlong( __longlong_as_double(oldval) < value ? __longlong_as_double(oldval) : value );
      }
    }

    __device__ __forceinline__ void atomicMax(float *address , float value) {
      int oldval, newval, readback;
      oldval = __float_as_int(*address);
      newval = __float_as_int( __int_as_float(oldval) > value ? __int_as_float(oldval) : value );
      while ( ( readback = atomicCAS( (int *) address , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __float_as_int( __int_as_float(oldval) > value ? __int_as_float(oldval) : value );
      }
    }

    __device__ __forceinline__ void atomicMax(double *address , double value) {
      unsigned long long oldval, newval, readback;
      oldval = __double_as_longlong(*address);
      newval = __double_as_longlong( __longlong_as_double(oldval) > value ? __longlong_as_double(oldval) : value );
      while ( ( readback = atomicCAS( (unsigned long long *) address , oldval , newval ) ) != oldval ) {
        oldval = readback;
        newval = __double_as_longlong( __longlong_as_double(oldval) > value ? __longlong_as_double(oldval) : value );
      }
    }
  #endif
  template <class FP> inline _YAKL void addAtomic(FP &x, FP const val) {
    if (target == targetCUDA) {
      #ifdef __NVCC__
        atomicAdd(&x,val);
      #endif
    } else if (target == targetCPUSerial) {
      x += val;
    }
  }

  template <class FP> inline _YAKL void minAtomic(FP &a, FP const b) {
    if (target == targetCUDA) {
      #ifdef __NVCC__
        atomicMin(&a,b);
      #endif
    } else if (target == targetCPUSerial) {
      a = a < b ? a : b;
    }
  }

  template <class FP> inline _YAKL void maxAtomic(FP &a, FP const b) {
    if (target == targetCUDA) {
      #ifdef __NVCC__
        atomicMax(&a,b);
      #endif
    } else if (target == targetCPUSerial) {
      a = a > b ? a : b;
    }
  }


}


#endif

