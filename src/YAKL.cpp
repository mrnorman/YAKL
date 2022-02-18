
#include "YAKL.h"


// This is where YAKL's statically scoped objects and Fortran-facing routines live.
// The goal is for these to be as minimal as possible.

namespace yakl {

  std::mutex yakl_mtx;  // Mutex lock for YAKL reference counting, allocation, and deallocation in threaded regions

  Gator pool;  // Pool allocator (manages multiple pools). Constructor does not allocate or depend on init order

  void * functorBuffer;  // To hold functors in device memory for CUDA (only large functors) and SYCL backends

  bool yakl_is_initialized = false;  // Determine if YAKL has been initialized

  // YAKL default allocaiton and deallocation functions on host and device. Init to give errors
  // when used before initializing the YAKL runtime with yakl::init()
  std::function<void *( size_t , char const *)> yaklAllocDeviceFunc = [] ( size_t bytes , char const *label ) -> void* {
    yakl_throw("ERROR: attempting memory alloc before calling yakl::init()");
    return nullptr;
  };
  std::function<void *( size_t , char const *)> yaklAllocHostFunc = [] ( size_t bytes , char const *label ) -> void* {
    yakl_throw("ERROR: attempting memory alloc before calling yakl::init()");
    return nullptr;
  };
  std::function<void ( void * , char const *)>  yaklFreeDeviceFunc  = [] ( void *ptr    , char const *label )          {
    yakl_throw("ERROR: attempting memory free before calling yakl::init()");
  };
  std::function<void ( void * , char const *)>  yaklFreeHostFunc  = [] ( void *ptr    , char const *label )          {
    yakl_throw("ERROR: attempting memory free before calling yakl::init()");
  };
}


// Fortran-facing routines

extern "C" void gatorInit() {
  yakl::init();
}

extern "C" void gatorFinalize() {
  yakl::finalize();
}

extern "C" void* gatorAllocate( size_t bytes ) {
  return yakl::yaklAllocDevice( bytes , "");
}

extern "C" void gatorDeallocate( void *ptr ) {
  yakl::yaklFreeDevice( ptr , "");
}


