
#pragma once

#include <iostream>
#include <iomanip>
#include <time.h>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cstring>
#include <vector>
#include "Gator.h"
#include "stdlib.h"

#ifdef YAKL_DEBUG
#include <stdexcept>
#include <sstream>
#include <string>
#endif

#ifdef __USE_CUDA__
  #define YAKL_LAMBDA [=] __device__
  #define YAKL_INLINE inline __host__ __device__
  #define YAKL_DEVICE inline __device__
  #include <cub/cub.cuh>
#elif defined(__USE_HIP__)
  #define YAKL_LAMBDA [=] __host__ __device__
  #define YAKL_INLINE inline __host__ __device__
  #define YAKL_DEVICE inline __device__
  #include "hip/hip_runtime.h"
  #include <hipcub/hipcub.hpp>
#else
  #define YAKL_LAMBDA [&]
  #define YAKL_INLINE inline
  #define YAKL_DEVICE inline
#endif

#ifdef _OPENMP45
#include <omp.h>
#endif

#ifdef _OPENACC
#include "openacc.h"
#endif


namespace yakl {

  typedef unsigned int index_t;

  template <class T> inline void yakl_throw(T &exc) {
    std::cout << "YAKL FATAL ERROR:\n";
    std::cout << exc << std::endl;
    throw exc;
  }

  // Memory space specifiers for YAKL Arrays
  int constexpr memDevice = 1;
  int constexpr memHost   = 2;
  int constexpr memStack  = 3;
  #if defined(__USE_CUDA__) || defined(__USE_HIP__)
    int constexpr memDefault = memDevice;
  #else
    int constexpr memDefault = memHost;
  #endif

  int constexpr styleC       = 1;
  int constexpr styleFortran = 2;
  int constexpr styleDefault = styleC;

  int constexpr COLON = std::numeric_limits<int>::min();
  int constexpr NOSPEC = std::numeric_limits<int>::min()+1;

  class Dims {
  public:
    int data[8];
    int rank;

    Dims() {rank = 0;}
    Dims(int i0) {
      data[0] = i0;
      rank = 1;
    }
    Dims(int i0, int i1) {
      data[0] = i0;
      data[1] = i1;
      rank = 2;
    }
    Dims(int i0, int i1, int i2) {
      data[0] = i0;
      data[1] = i1;
      data[2] = i2;
      rank = 3;
    }
    Dims(int i0, int i1, int i2, int i3) {
      data[0] = i0;
      data[1] = i1;
      data[2] = i2;
      data[3] = i3;
      rank = 4;
    }
    Dims(int i0, int i1, int i2, int i3, int i4) {
      data[0] = i0;
      data[1] = i1;
      data[2] = i2;
      data[3] = i3;
      data[4] = i4;
      rank = 5;
    }
    Dims(int i0, int i1, int i2, int i3, int i4, int i5) {
      data[0] = i0;
      data[1] = i1;
      data[2] = i2;
      data[3] = i3;
      data[4] = i4;
      data[5] = i5;
      rank = 6;
    }
    Dims(int i0, int i1, int i2, int i3, int i4, int i5, int i6) {
      data[0] = i0;
      data[1] = i1;
      data[2] = i2;
      data[3] = i3;
      data[4] = i4;
      data[5] = i5;
      data[6] = i6;
      rank = 7;
    }
    Dims(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7) {
      data[0] = i0;
      data[1] = i1;
      data[2] = i2;
      data[3] = i3;
      data[4] = i4;
      data[5] = i5;
      data[6] = i6;
      data[7] = i7;
      rank = 8;
    }

    int size() const {
      return rank;
    }
  };


  // Size of the buffer to hold large functors for the CUDA backend to avoid exceeding the max stack frame
  int constexpr functorBufSize = 1024*128;
  // Buffer to hold large functors for the CUDA backend to avoid exceeding the max stack frame
  extern void *functorBuffer;


  // Pool allocator object
  extern Gator pool;

  // YAKL allocator and deallocator
  extern std::function<void *( size_t , char const *)> yaklAllocDeviceFunc;
  extern std::function<void ( void * , char const *)>  yaklFreeDeviceFunc;

  // YAKL allocator and deallocator
  extern std::function<void *( size_t , char const *)> yaklAllocHostFunc;
  extern std::function<void ( void * , char const *)>  yaklFreeHostFunc;

  extern bool yakl_is_initialized;

  #ifdef __USE_HIP__
    YAKL_INLINE void *yaklAllocDevice( size_t bytes , char const *label ) { return yaklAllocDeviceFunc(bytes,label); }
    YAKL_INLINE void yaklFreeDevice( void *ptr , char const *label ) { yaklFreeDeviceFunc(ptr,label); }
    YAKL_INLINE void *yaklAllocHost( size_t bytes , char const *label ) { return yaklAllocHostFunc(bytes,label); }
    YAKL_INLINE void yaklFreeHost( void *ptr , char const *label ) { yaklFreeHostFunc(ptr,label); }
  #else
    void *yaklAllocDevice( size_t bytes , char const *label );
    void yaklFreeDevice( void *ptr , char const *label );
    void *yaklAllocHost( size_t bytes , char const *label );
    void yaklFreeHost( void *ptr , char const *label );
  #endif

    


  // [S]tatic (compile-time) Array [B]ounds (templated)
  // It's only used for Fortran, so it takes on Fortran defaults
  // with lower bound default to 1
  template <int L, int U=-999> class SB {
  public:
    SB() = delete;
  };

  // Fortran list of static bounds
  template <class T, class B0, class B1=SB<1,1>, class B2=SB<1,1>, class B3=SB<1,1>> class FSPEC {
  public:
    FSPEC() = delete;
  };

  // C list of static dimension sizes
  template <class T, unsigned D0, unsigned D1=1, unsigned D2=1, unsigned D3=1> class CSPEC {
  public:
    CSPEC() = delete;
  };


  // Dynamic (runtime) Array Bounds
  class Bnd {
  public:
    int l, u;
    Bnd(                  ) { l = 1   ; u = 1   ; }
    Bnd(          int u_in) { l = 1   ; u = u_in; }
    Bnd(int l_in, int u_in) { l = l_in; u = u_in; }
  };



  // Block the CPU code until the device code and data transfers are all completed
  inline void fence() {
    #ifdef __USE_CUDA__
      cudaDeviceSynchronize();
    #endif
    #ifdef __USE_HIP__
      hipDeviceSynchronize();
    #endif
  }


  inline bool isInitialized() {
    return yakl_is_initialized;
  }


#include "YAKL_init.h"


  inline void finalize() {
    yakl_is_initialized = false;
    size_t hwm = pool.highWaterMark();
    if        (hwm >= 1024*1024*1024) {
      std::cout << "Memory high water mark: " << (double) hwm / (double) (1024*1024*1024) << " GB\n";
    } else if (hwm >= 1024*1024     ) {
      std::cout << "Memory high water mark: " << (double) hwm / (double) (1024*1024     ) << " MB\n";
    } else if (hwm >= 1024          ) {
      std::cout << "Memory high water mark: " << (double) hwm / (double) (1024          ) << " KB\n";
    }
    pool.finalize();
    #if defined(__USE_CUDA__)
      cudaFree(functorBuffer);
    #endif
  }


#include "YAKL_parallel_for.h"


#include "YAKL_reductions.h"


#include "YAKL_atomics.h"


#include "YAKL_random.h"


template <class T> YAKL_INLINE constexpr T max(T a, T b) { return a>b? a : b; }
template <class T> YAKL_INLINE constexpr T min(T a, T b) { return a<b? a : b; }
template <class T> YAKL_INLINE constexpr T abs(T a) { return a>0? a : -a; }


#include "Array.h"


#include "ScalarLiveOut.h"


#include "FortranIntrinsics.h"


template <class T, int rank, int myMem, int myStyle> void memset( Array<T,rank,myMem,myStyle> &arr , T val ) {
  if (myMem == memDevice) {
    c::parallel_for( arr.totElems() , YAKL_LAMBDA (int i) {
      arr.myData[i] = val;
    });
  } else if (myMem == memHost) {
    for (size_t i = 0; i < arr.totElems(); i++) {
      arr.myData[i] = val;
    }
  }
}


}



