
#include "YAKL.h"


// This is where YAKL's statically scoped objects and Fortran-facing routines live.
// The goal is for these to be as minimal as possible.

namespace yakl {

  std::mutex yakl_mtx;  // Mutex lock for YAKL reference counting, allocation, and deallocation in threaded regions

  Gator pool;  // Pool allocator (manages multiple pools). Constructor does not allocate or depend on init order

  void * functorBuffer = nullptr;  // To hold functors in device memory for CUDA (only large functors) and SYCL backends

  bool yakl_is_initialized = false;  // Determine if YAKL has been initialized

  Toney timer;

  std::function<void ()> timer_init_func = [] () {
    yakl_throw("ERROR: attempting to call the yakl::timer_init(); before calling yakl::init()");
  };
  std::function<void ()> timer_finalize_func = [] () {
    yakl_throw("ERROR: attempting to call the yakl::timer_finalize(); before calling yakl::init()");
  };
  std::function<void (char const *)> timer_start_func = [] (char const *label) {
    yakl_throw("ERROR: attempting to call the yakl::timer_start(); before calling yakl::init()");
  };
  std::function<void (char const *)> timer_stop_func = [] (char const * label) {
    yakl_throw("ERROR: attempting to call the yakl::timer_stop(); before calling yakl::init()");
  };
  std::function<void *( size_t , char const *)> alloc_device_func = [] ( size_t bytes , char const *label ) -> void* {
    yakl_throw("ERROR: attempting memory alloc before calling yakl::init()");
    return nullptr;
  };
  std::function<void *( size_t , char const *)> alloc_host_func   = [] ( size_t bytes , char const *label ) -> void* {
    yakl_throw("ERROR: attempting memory alloc before calling yakl::init()");
    return nullptr;
  };
  std::function<void ( void * , char const *)>  free_device_func  = [] ( void *ptr    , char const *label )          {
    yakl_throw("ERROR: attempting memory free before calling yakl::init()");
  };
  std::function<void ( void * , char const *)>  free_host_func    = [] ( void *ptr    , char const *label )          {
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
  return yakl::alloc_device( bytes , "gatorAllocate");
}

extern "C" void gatorDeallocate( void *ptr ) {
  yakl::free_device( ptr , "gatorDeallocate");
}


