/**
 * @file
 * YAKL finalization routine
 */

#pragma once
// Included by YAKL.h

__YAKL_NAMESPACE_WRAPPER_BEGIN__
namespace yakl {

  /**
   * @brief Finalize the YAKL runtime.
   * @details Best practice is to call yakl::isInitialized() to ensure the YAKL runtime
   * is initialized before calling this routine. That said, this routine *does* check to ensure the runtime
   * is initialized for you. THREAD SAFE!
   */
  inline void finalize() {
    // We don't know what happens in finalize_callbacks, so to be safe, let's use a unique mutex here.
    get_yakl_instance().yakl_final_mtx.lock();

    // Only finalize if YAKL's already initialized
    if ( isInitialized() ) {
      fence();  // Make sure all device work is done before we start freeing pool memory

      for (int i=0; i < get_yakl_instance().finalize_callbacks.size(); i++) {
        get_yakl_instance().finalize_callbacks[i]();
      }

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

    // We don't know what happens in finalize_callbacks, so to be safe, let's use a unique mutex here.
    get_yakl_instance().yakl_final_mtx.unlock();
  }

  /** @brief Add a host-only callback to be called just before YAKL finalizes. This is useful for ensuring allcoated
   *         global variables are always deallocated before YAKL finalization.
   *  @details YAKL calls a fence() beforehand and does nothing else until all finalize callbacks are completed. One
   *           thing this protects against in particular is the pool being finalized before all variables ar
   *           deallocated.
   *  @param  callback   A void function with no parameters that does finalization / cleanup / deallocation work
   *                     before YAKL finalize.
   */
  inline void register_finalize_callback( std::function<void ()> callback ) { get_yakl_instance().finalize_callbacks.push_back(callback); }
}
__YAKL_NAMESPACE_WRAPPER_END__


