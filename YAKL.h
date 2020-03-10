
#pragma once

#include <iostream>
#include <iomanip>
#include <time.h>
#include <algorithm>
#include <limits>
#include <cmath>
#include "BuddyAllocator.h"
#include "stdlib.h"

#ifdef ARRAY_DEBUG
#include <stdexcept>
#include <sstream>
#include <string>
#endif

#ifdef __USE_CUDA__
  #define YAKL_LAMBDA [=] __device__
  #define YAKL_INLINE inline __host__ __device__
  #include <cub/cub.cuh>
#elif defined(__USE_HIP__)
  #define YAKL_LAMBDA [=] __host__ __device__
  #define YAKL_INLINE inline __host__ __device__
  #include "hip/hip_runtime.h"
  #include <hipcub/hipcub.hpp>
#else
  #define YAKL_LAMBDA [&]
  #define YAKL_INLINE inline
#endif

#ifdef _OPENMP45
#include <omp.h>
#endif

#ifdef _OPENACC
#include "openacc.h"
#endif


namespace yakl {

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

  struct Dims {
    int data[8];
  };


  // Size of the buffer to hold large functors for the CUDA backend to avoid exceeding the max stack frame
  int constexpr functorBufSize = 1024*128;
  // Buffer to hold large functors for the CUDA backend to avoid exceeding the max stack frame
  extern void *functorBuffer;


  // Pool allocator object
  extern BuddyAllocator pool;

  // YAKL allocator and deallocator
  extern std::function<void *( size_t )> yaklAllocDevice;
  extern std::function<void ( void * )>  yaklFreeDevice;

  // YAKL allocator and deallocator
  extern std::function<void *( size_t )> yaklAllocHost;
  extern std::function<void ( void * )>  yaklFreeHost;


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
  template <class T, size_t D0, size_t D1=1, size_t D2=1, size_t D3=1> class CSPEC {
  public:
    CSPEC() = delete;
  };


  // Dynamic (runtime) Array Bounds
  class Bnd {
  public:
    int l, u;
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


#include "YAKL_init.h"


  inline void finalize() {
    pool = BuddyAllocator();
    #if defined(__USE_CUDA__)
      cudaFree(functorBuffer);
    #endif
  }


#include "YAKL_parallel_for.h"


#include "YAKL_reductions.h"


#include "YAKL_atomics.h"


#include "YAKL_random.h"


#include "YAKL_fft.h"


template <class T> YAKL_INLINE T max(T a, T b) { return a>b? a : b; }
template <class T> YAKL_INLINE T min(T a, T b) { return a<b? a : b; }
template <class T> YAKL_INLINE T abs(T a) { return a>0? a : -a; }


#include "Array.h"


}



