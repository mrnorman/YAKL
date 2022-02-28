
#pragma once
// Included by YAKL.h
// Inside the yakl namespace

// Free the functorBuffer, free the memory pools, reset global std::functions for alloc and free
inline void finalize() {
  yakl_mtx.lock();

  // Only finalize if YAKL's already initialized
  if ( isInitialized() ) {
    fence();  // Make sure all device work is done before we start freeing pool memory

    // Free the pools
    pool.finalize();

    // Free functorBuffer
    #ifdef YAKL_ARCH_CUDA
      cudaFree(functorBuffer);
      check_last_error();
    #endif
    #if defined(YAKL_ARCH_SYCL)
      sycl::free(functorBuffer, sycl_default_stream());
      check_last_error();
    #endif

    yakl_is_initialized = false;

    // Finalize the timers
    #if defined(YAKL_PROFILE) || defined(YAKL_AUTO_PROFILE)
      if (yakl_masterproc()) {
        GPTLpr_file("");
      }
      GPTLpr_file("yakl_timer_output.txt");
    #endif

    // Reset the global std::functions for host and device alloc and free
    yaklAllocDeviceFunc = [] ( size_t bytes , char const *label ) -> void* {
      yakl_throw("ERROR: attempting memory alloc before calling yakl::init()");
      return nullptr;
    };
    yaklFreeDeviceFunc  = [] ( void *ptr    , char const *label )          {
      yakl_throw("ERROR: attempting memory free before calling yakl::init()");
    };
    yaklAllocHostFunc = [] ( size_t bytes , char const *label ) -> void* {
      yakl_throw("ERROR: attempting memory alloc before calling yakl::init()");
      return nullptr;
    };
    yaklFreeHostFunc  = [] ( void *ptr    , char const *label )          {
      yakl_throw("ERROR: attempting memory free before calling yakl::init()");
    };

  } else {

    std::cerr << "WARNING: Calling yakl::finalize() when YAKL is not initialized. ";
    std::cerr << "This might mean you've called yakl::finalize() more than once.\n";

  }

  yakl_is_initialized = false;

  yakl_mtx.unlock();
}

