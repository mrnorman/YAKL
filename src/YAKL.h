
#pragma once

#include "YAKL_header.h"
#include "YAKL_defines.h"

namespace yakl {
  using std::cos;
  using std::sin;
  using std::pow;
  using std::min;
  using std::max;
  using std::abs;

  // For thread safety in YAKL Array reference counters
  extern std::mutex yakl_mtx;

  // YAKL allocator and deallocator on host and device as std::function's
  extern std::function<void *( size_t , char const *)> yaklAllocHost;
  extern std::function<void *( size_t , char const *)> yaklAllocDevice;
  extern std::function<void ( void * , char const *)>  yaklFreeHost;
  extern std::function<void ( void * , char const *)>  yaklFreeDevice;

  inline void set_host_allocator    ( std::function<void *(size_t,char const *)> func ) { yaklAllocHost   = func; }
  inline void set_device_allocator  ( std::function<void *(size_t,char const *)> func ) { yaklAllocDevice = func; }
  inline void set_host_deallocator  ( std::function<void (void *,char const *)>  func ) { yaklFreeHost    = func; }
  inline void set_device_deallocator( std::function<void (void *,char const *)>  func ) { yaklFreeDevice  = func; }

  // YAKL's default allocation, free, mutex lock, and mutex unlock routines.
  inline void yakl_mtx_lock  () { yakl_mtx.lock  (); }
  inline void yakl_mtx_unlock() { yakl_mtx.unlock(); }

  // Bool and function to see if YAKL has been initialized yet
  extern bool yakl_is_initialized;
  inline bool isInitialized() { return yakl_is_initialized; }

  // SYCL and CUDA have situations where functors need to be explicitly copied to buffers
  // and then run from device memory
  #if defined(YAKL_ARCH_CUDA) || defined (YAKL_ARCH_SYCL)
    // Size of the buffer to hold large functors for the CUDA and SYCL backends to avoid exceeding the max stack frame
    int constexpr functorBufSize = 1024*128;
    // Buffer to hold large functors for the CUDA and SYCL backends to avoid exceeding the max stack frame
    extern void *functorBuffer;
  #endif

  // Type for indexing. Rarely if ever is size_t going to be needed
  typedef unsigned int index_t;
  index_t constexpr INDEX_MAX = std::numeric_limits<index_t>::max();

  // Labels for memory spaces. Only memDevice and memHost are expected to be used explicitly by the user
  int constexpr memDevice = 1;
  int constexpr memHost   = 2;
  int constexpr memStack  = 3;
  #if defined(YAKL_ARCH_CUDA) || defined(YAKL_ARCH_HIP) || defined(YAKL_ARCH_SYCL)
    int constexpr memDefault = memDevice;
  #else
    int constexpr memDefault = memHost;
  #endif

  // Labels for Array styles. C has zero-based indexing with the last index varying the fastest.
  // Fortran has 1-based indexing with arbitrary lower bounds and the index varying the fastest.
  int constexpr styleC       = 1;
  int constexpr styleFortran = 2;
  int constexpr styleDefault = styleC;

  int constexpr COLON = std::numeric_limits<int>::min(); // Label for the ":" from Fortrna array slicing
}

#include "YAKL_error.h"
#include "YAKL_sycldevice.h"
#include "YAKL_LaunchConfig.h"
#include "YAKL_fence.h"
#include "YAKL_simd.h"
#include "YAKL_alloc_free.h"
#include "YAKL_memory_pool.h"
#include "YAKL_timers.h"
#include "YAKL_init.h"
#include "YAKL_finalize.h"
#include "YAKL_parallel_for.h"
#include "YAKL_reductions.h"
#include "YAKL_atomics.h"
#include "YAKL_random.h"
#include "YAKL_mem_transfers.h"
#include "YAKL_Array.h"
#include "YAKL_ScalarLiveOut.h"
#include "YAKL_componentwise.h"
#include "YAKL_Intrinsics.h"
#include "YAKL_memset.h"

