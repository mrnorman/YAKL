
#include "YAKL.h"


// This is where YAKL's statically scoped objects and Fortran-facing routines live.
// The goal is for these to be as minimal as possible.

namespace yakl {

  std::mutex yakl_mtx;  // Mutex lock for YAKL reference counting, allocation, and deallocation in threaded regions

  Gator pool;  // Pool allocator (manages multiple pools). Constructor does not allocate or depend on init order

  void * functorBuffer = nullptr;  // To hold functors in device memory for CUDA (only large functors) and SYCL backends

  bool yakl_is_initialized = false;  // Determine if YAKL has been initialized

  Toney timer;

  std::function<void ()> timer_init = [] () {
    yakl_throw("ERROR: attempting to call the yakl::timer_init(); before calling yakl::init()");
  };
  std::function<void ()> timer_finalize = [] () {
    yakl_throw("ERROR: attempting to call the yakl::timer_finalize(); before calling yakl::init()");
  };
  std::function<void (char const *)> timer_start = [] (char const *label) {
    yakl_throw("ERROR: attempting to call the yakl::timer_start(); before calling yakl::init()");
  };
  std::function<void (char const *)> timer_stop = [] (char const * label) {
    yakl_throw("ERROR: attempting to call the yakl::timer_stop(); before calling yakl::init()");
  };
  std::function<void *( size_t , char const *)> yaklAllocDevice = [] ( size_t bytes , char const *label ) -> void* {
    yakl_throw("ERROR: attempting memory alloc before calling yakl::init()");
    return nullptr;
  };
  std::function<void *( size_t , char const *)> yaklAllocHost   = [] ( size_t bytes , char const *label ) -> void* {
    yakl_throw("ERROR: attempting memory alloc before calling yakl::init()");
    return nullptr;
  };
  std::function<void ( void * , char const *)>  yaklFreeDevice  = [] ( void *ptr    , char const *label )          {
    yakl_throw("ERROR: attempting memory free before calling yakl::init()");
  };
  std::function<void ( void * , char const *)>  yaklFreeHost    = [] ( void *ptr    , char const *label )          {
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
  return yakl::yaklAllocDevice( bytes , "gatorAllocate");
}

extern "C" void gatorDeallocate( void *ptr ) {
  yakl::yaklFreeDevice( ptr , "gatorDeallocate");
}


