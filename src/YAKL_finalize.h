/**
 * @file
 * YAKL finalization routine
 */

#pragma once
// Included by YAKL.h

namespace yakl {

  /**
   * @brief Finalize the YAKL runtime.
   * @details Best practice is to call yakl::isInitialized() to ensure the YAKL runtime
   * is initialized before calling this routine. That said, this routine *does* check to ensure the runtime
   * is initialized for you. THREAD SAFE!
   */
  inline void finalize() {
    yakl_mtx_lock();

    // Only finalize if YAKL's already initialized
    if ( isInitialized() ) {
      fence();  // Make sure all device work is done before we start freeing pool memory

      #if defined(YAKL_ARCH_HIP)
        rocfft_cleanup();
      #endif

      // Free the pools
      get_yakl_instance().pool.finalize();

      get_yakl_instance().yakl_is_initialized = false;

      // Finalize the timers
      #if defined(YAKL_PROFILE)
        timer_finalize();
      #endif

      get_yakl_instance().timer_init_func = [] () {
        yakl_throw("ERROR: attempting to call the yakl::timer_init(); before calling yakl::init()");
      };
      get_yakl_instance().timer_finalize_func = [] () {
        yakl_throw("ERROR: attempting to call the yakl::timer_finalize(); before calling yakl::init()");
      };
      get_yakl_instance().timer_start_func = [] (char const *label) {
        yakl_throw("ERROR: attempting to call the yakl::timer_start(); before calling yakl::init()");
      };
      get_yakl_instance().timer_stop_func = [] (char const * label) {
        yakl_throw("ERROR: attempting to call the yakl::timer_stop(); before calling yakl::init()");
      };
      get_yakl_instance().alloc_device_func = [] ( size_t bytes , char const *label ) -> void* {
        yakl_throw("ERROR: attempting memory alloc before calling yakl::init()");
        return nullptr;
      };
      get_yakl_instance().free_device_func  = [] ( void *ptr    , char const *label )          {
        yakl_throw("ERROR: attempting memory free before calling yakl::init()");
      };

      get_yakl_instance().device_allocators_are_default = false;
      get_yakl_instance().pool_enabled = false;

    } else {

      std::cerr << "WARNING: Calling yakl::finalize() when YAKL is not initialized. ";
      std::cerr << "This might mean you've called yakl::finalize() more than once.\n";

    }

    get_yakl_instance().yakl_is_initialized = false;

    yakl_mtx_unlock();
  }
}


