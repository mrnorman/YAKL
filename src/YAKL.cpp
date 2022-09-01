
#include "YAKL.h"


// This is where YAKL's statically scoped objects and Fortran-facing routines live.
// The goal is for these to be as minimal as possible.

namespace yakl {

  /** @private */
  std::mutex yakl_mtx;  // Mutex lock for YAKL reference counting, allocation, and deallocation in threaded regions

  /** @private */
  Gator pool;  // Pool allocator (manages multiple pools). Constructor does not allocate or depend on init order

  /** @private */
  bool yakl_is_initialized = false;  // Determine if YAKL has been initialized

  /** @private */
  Toney timer;

  /** @private */
  std::function<void ()> timer_init_func = [] () {
    yakl_throw("ERROR: attempting to call the yakl::timer_init(); before calling yakl::init()");
  };
  /** @private */
  std::function<void ()> timer_finalize_func = [] () {
    yakl_throw("ERROR: attempting to call the yakl::timer_finalize(); before calling yakl::init()");
  };
  /** @private */
  std::function<void (char const *)> timer_start_func = [] (char const *label) {
    yakl_throw("ERROR: attempting to call the yakl::timer_start(); before calling yakl::init()");
  };
  /** @private */
  std::function<void (char const *)> timer_stop_func = [] (char const * label) {
    yakl_throw("ERROR: attempting to call the yakl::timer_stop(); before calling yakl::init()");
  };
  /** @private */
  std::function<void *( size_t , char const *)> alloc_device_func = [] ( size_t bytes , char const *label ) -> void* {
    yakl_throw("ERROR: attempting memory alloc before calling yakl::init()");
    return nullptr;
  };
  /** @private */
  std::function<void ( void * , char const *)>  free_device_func  = [] ( void *ptr    , char const *label )          {
    yakl_throw("ERROR: attempting memory free before calling yakl::init()");
  };

  /** @private */
  bool device_allocators_are_default  = false;

  /** @private */
  bool pool_enabled                   = false;
}


// Fortran-facing routines

/** @brief Fortran YAKL initialization call */
extern "C" void gatorInit() {
  yakl::init();
}

/** @brief Fortran YAKL finalization call */
extern "C" void gatorFinalize() {
  yakl::finalize();
}

/** @brief Fortran YAKL device allocation call */
extern "C" void* gatorAllocate( size_t bytes ) {
  return yakl::alloc_device( bytes , "gatorAllocate");
}

/** @brief Fortran YAKL device free call */
extern "C" void gatorDeallocate( void *ptr ) {
  yakl::free_device( ptr , "gatorDeallocate");
}


