
#ifndef _YAKL_H_
#define _YAKL_H_

#include <iostream>
#include <algorithm>
#include <vector>
#include <limits>
#include "BuddyAllocator.h"

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
  #if defined(__USE_CUDA__) || defined(__USE_HIP__)
    int constexpr memDefault = memDevice;
  #else
    int constexpr memDefault = memHost;
  #endif

  int constexpr COLON = std::numeric_limits<int>::min();
  int constexpr NOSPEC = std::numeric_limits<int>::min()+1;


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


  template <int L, int U> class bnd {
  public:
    bnd() = delete;
    static constexpr int l() { return L; }
    static constexpr int u() { return U; }
  };


  // Must be constructed on the CPU, but operator() and niter() are GPU-callable
  class Bounds {
    int numIterations;
    int data[8][3];
  public:
    Bounds(std::vector<int> b0 ,
           std::vector<int> b1 = {0,0} ,
           std::vector<int> b2 = {0,0} ,
           std::vector<int> b3 = {0,0} ,
           std::vector<int> b4 = {0,0} ,
           std::vector<int> b5 = {0,0} ,
           std::vector<int> b6 = {0,0} ,
           std::vector<int> b7 = {0,0} ) {
      // Store bounds
      // LOOP BEGINNING      LOOP END              LOOP STRIDE (only if specified)
      data[0][0] = b0[0];   data[0][1] = b0[1];   data[0][2] = b0.size() >= 3 ? b0[2] : 1;
      data[1][0] = b1[0];   data[1][1] = b1[1];   data[1][2] = b1.size() >= 3 ? b0[2] : 1;
      data[2][0] = b2[0];   data[2][1] = b2[1];   data[2][2] = b2.size() >= 3 ? b0[2] : 1;
      data[3][0] = b3[0];   data[3][1] = b3[1];   data[3][2] = b3.size() >= 3 ? b0[2] : 1;
      data[4][0] = b4[0];   data[4][1] = b4[1];   data[4][2] = b4.size() >= 3 ? b0[2] : 1;
      data[5][0] = b5[0];   data[5][1] = b5[1];   data[5][2] = b5.size() >= 3 ? b0[2] : 1;
      data[6][0] = b6[0];   data[6][1] = b6[1];   data[6][2] = b6.size() >= 3 ? b0[2] : 1;
      data[7][0] = b7[0];   data[7][1] = b7[1];   data[7][2] = b7.size() >= 3 ? b0[2] : 1;

      // Process bounds
      numIterations = 1;
      for (int i=0; i<8; i++) {
        // Store the dimension size in data[*][1]. Inherent floor operation with integer division below
        data[i][1] = (data[i][1] - data[i][0] + 1) / data[i][2];
        numIterations *= data[i][1];   // Keep track of total nested loop iterations
      }
    }
    YAKL_INLINE int operator() (int j, int i) const {
      return data[j][i];
    }
    YAKL_INLINE int nIter() const {
      return numIterations;
    }
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


#include "YAKL_unpack.h"


#include "YAKL_parallel_for_c.h"


#include "YAKL_parallel_for_fortran.h"


#include "YAKL_reductions.h"


#include "YAKL_atomics.h"


}


#endif

