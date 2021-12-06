
#pragma once

#include "YAKL_header.h"

namespace yakl {

  extern std::mutex yakl_mtx;

  typedef unsigned int index_t;
  index_t constexpr INDEX_MAX = std::numeric_limits<index_t>::max();

  // Memory space specifiers for YAKL Arrays
  int constexpr memDevice = 1;
  int constexpr memHost   = 2;
  int constexpr memStack  = 3;
  #if defined(YAKL_ARCH_CUDA) || defined(YAKL_ARCH_HIP) || defined(YAKL_ARCH_SYCL) || defined(YAKL_ARCH_OPENMP45)
    int constexpr memDefault = memDevice;
  #else
    int constexpr memDefault = memHost;
  #endif

  int constexpr styleC       = 1;
  int constexpr styleFortran = 2;
  int constexpr styleDefault = styleC;

  int constexpr COLON = std::numeric_limits<int>::min();
  int constexpr NOSPEC = std::numeric_limits<int>::min()+1;


  #if defined(YAKL_ARCH_CUDA) || defined (YAKL_ARCH_SYCL)
    // Size of the buffer to hold large functors for the CUDA and SYCL backends to avoid exceeding the max stack frame
    int constexpr functorBufSize = 1024*128;
    // Buffer to hold large functors for the CUDA and SYCL backends to avoid exceeding the max stack frame
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


  #if defined(YAKL_ARCH_SYCL)
    YAKL_INLINE void *yaklAllocDevice( size_t bytes , char const *label ) { return yaklAllocDeviceFunc(bytes,label); }
    YAKL_INLINE void yaklFreeDevice( void *ptr , char const *label ) { yaklFreeDeviceFunc(ptr,label); }
    YAKL_INLINE void *yaklAllocHost( size_t bytes , char const *label ) { return yaklAllocHostFunc(bytes,label); }
    YAKL_INLINE void yaklFreeHost( void *ptr , char const *label ) { yaklFreeHostFunc(ptr,label); }
    YAKL_INLINE void yakl_mtx_lock()   { yakl_mtx.lock(); }
    YAKL_INLINE void yakl_mtx_unlock() { yakl_mtx.unlock(); }
  #else   // HIP AND CUDA
    inline void *yaklAllocDevice( size_t bytes , char const *label ) { return yaklAllocDeviceFunc(bytes,label); }
    inline void yaklFreeDevice( void *ptr , char const *label ) { yaklFreeDeviceFunc(ptr,label); }
    inline void *yaklAllocHost( size_t bytes , char const *label ) { return yaklAllocHostFunc(bytes,label); }
    inline void yaklFreeHost( void *ptr , char const *label ) { yaklFreeHostFunc(ptr,label); }
    inline void yakl_mtx_lock()   { yakl_mtx.lock(); }
    inline void yakl_mtx_unlock() { yakl_mtx.unlock(); }
  #endif


  // Block the CPU code until the device code and data transfers are all completed
  inline void fence() {
    #ifdef YAKL_ARCH_CUDA
      cudaDeviceSynchronize();
      check_last_error();
    #endif
    #ifdef YAKL_ARCH_HIP
      hipDeviceSynchronize();
      check_last_error();
    #endif
    #ifdef YAKL_ARCH_SYCL
      sycl_default_stream().wait();
      check_last_error();
    #endif
  }

  #include "YAKL_timers.h"


  inline bool isInitialized() {
    return yakl_is_initialized;
  }


#include "YAKL_init.h"


  inline void finalize() {
    yakl_mtx.lock();

    if ( isInitialized() ) {
      pool.finalize();
      #ifdef YAKL_ARCH_CUDA
        cudaFree(functorBuffer);
        check_last_error();
      #endif
      #if defined(YAKL_ARCH_SYCL)
        sycl::free(functorBuffer, sycl_default_stream());
        check_last_error();
      #endif
      yakl_is_initialized = false;
      #if defined(YAKL_PROFILE) || defined(YAKL_AUTO_PROFILE)
        GPTLpr_file("");
        GPTLpr_file("yakl_timer_output.txt");
      #endif
    } else {
      std::cerr << "WARNING: Calling yakl::finalize() when YAKL is not initialized. ";
      std::cerr << "This might mean you've called yakl::finalize() more than once.\n";
    }
    yakl_is_initialized = false;

    yakl_mtx.unlock();
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


#include "YAKL_mem_transfers.h"


#include "YAKL_Array.h"


#include "YAKL_ScalarLiveOut.h"


#include "YAKL_Intrinsics.h"


  /////////////////////////////////////////////////
  // memset
  /////////////////////////////////////////////////
  template <class T, int rank, int myMem, int myStyle, class I>
  void memset( Array<T,rank,myMem,myStyle> &arr , I val ) {
    if (myMem == memDevice) {
      c::parallel_for( arr.totElems() , YAKL_LAMBDA (int i) {
        arr.myData[i] = val;
      });
      #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
        fence();
      #endif
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
