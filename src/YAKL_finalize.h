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
    // We don't know what happens in finalize_callbacks, so to be safe, let's use a unique mutex here.
    get_yakl_instance().yakl_final_mtx.lock();

    // Only finalize if YAKL's already initialized
    if ( isInitialized() ) {

      Kokkos::fence();  // Make sure all device work is done before we start freeing pool memory
      get_yakl_instance().pool.finalize();
      get_yakl_instance().yakl_is_initialized = false;
      #if defined(YAKL_PROFILE)
        timer_finalize();
      #endif
      get_yakl_instance().pool_enabled = false;
    } else {
      #ifdef KOKKOS_ENABLE_DEBUG
        std::cerr << "WARNING: Calling yakl::finalize() when YAKL is not initialized. ";
        std::cerr << "This might mean you've called yakl::finalize() more than once.\n";
      #endif
    }
    get_yakl_instance().yakl_is_initialized = false;
    // We don't know what happens in finalize_callbacks, so to be safe, let's use a unique mutex here.
    get_yakl_instance().yakl_final_mtx.unlock();
  }

}


