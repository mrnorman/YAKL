
#pragma once

inline void finalize() {
  yakl_mtx.lock();

  if ( isInitialized() ) {
    fence();
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
      timer_finalize();
    #endif
    // YAKL allocator and deallocator
    yaklAllocDeviceFunc = [] ( size_t bytes , char const *label ) -> void* {
      yakl_throw("ERROR: attempting memory alloc before calling yakl::init()");
      return nullptr;
    };
    yaklFreeDeviceFunc  = [] ( void *ptr    , char const *label )          {
      yakl_throw("ERROR: attempting memory free before calling yakl::init()");
    };
    // YAKL allocator and deallocator
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

