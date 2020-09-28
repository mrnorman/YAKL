
#pragma once

#include "YAKL_header.h"

namespace yakl {

  typedef unsigned int index_t;


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


  #ifdef __USE_CUDA__
    // Size of the buffer to hold large functors for the CUDA backend to avoid exceeding the max stack frame
    int constexpr functorBufSize = 1024*128;
    // Buffer to hold large functors for the CUDA backend to avoid exceeding the max stack frame
    extern void *functorBuffer;
  #endif


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


  // Block the CPU code until the device code and data transfers are all completed
  inline void fence() {
    #ifdef __USE_CUDA__
      cudaDeviceSynchronize();
      check_last_error();
    #endif
    #ifdef __USE_HIP__
      hipDeviceSynchronize();
      check_last_error();
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
      check_last_error();
    #endif
  }


#include "YAKL_parallel_for.h"


#include "YAKL_reductions.h"


#include "YAKL_atomics.h"


#include "YAKL_random.h"


  /////////////////////////////////////////////////
  // min, max, abs
  /////////////////////////////////////////////////
  template <class T> YAKL_INLINE constexpr T max(T a, T b) { return a>b? a : b; }
  template <class T> YAKL_INLINE constexpr T min(T a, T b) { return a<b? a : b; }
  template <class T> YAKL_INLINE constexpr T abs(T a) { return a>0? a : -a; }


#include "Array.h"


#include "ScalarLiveOut.h"


#include "Intrinsics.h"


  /////////////////////////////////////////////////
  // memset
  /////////////////////////////////////////////////
  template <class T, int rank, int myMem, int myStyle, class I>
  void memset( Array<T,rank,myMem,myStyle> &arr , I val ) {
    if (myMem == memDevice) {
      c::parallel_for( arr.totElems() , YAKL_LAMBDA (int i) {
        arr.myData[i] = val;
      });
    } else if (myMem == memHost) {
      for (index_t i = 0; i < arr.totElems(); i++) {
        arr.myData[i] = val;
      }
    }
  }
  template <class T, int rank, class B0, class B1, class B2, class B3, class I>
  void memset( FSArray<T,rank,B0,B1,B2,B3> &arr , I val ) {
    for (index_t i = 0; i < arr.totElems(); i++) {
      arr.myData[i] = val;
    }
  }
  template <class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3, class I>
  void memset( SArray<T,rank,D0,D1,D2,D3> &arr , I val ) {
    for (index_t i = 0; i < arr.totElems(); i++) {
      arr.myData[i] = val;
    }
  }

}



