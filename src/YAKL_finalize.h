
#pragma once
// Included by YAKL.h

namespace yakl {

  /** Finalize YAKL from the controlling host thread after all application host-threaded regions have completed.
    * All host allocation/deallocation calls must be complete and all DeviceSpace owners must already be destroyed.
    */
  inline void finalize() {
    if constexpr (kokkos_debug) {
      if (!Kokkos::is_initialized()) Kokkos::abort("ERROR: yakl::finalize called after Kokkos::finalize");
    }
    // Only finalize if YAKL's already initialized
    if ( get_yakl_instance().is_initialized() ) {
      Kokkos::fence();  // Make sure all device work is done before we start freeing pool memory
      if (get_yakl_instance().num_device_allocations != 0) {
        Kokkos::abort("ERROR: yakl::finalize called while YAKL device allocations are still alive");
      }
      timer_print();
      get_yakl_instance().timer.clear();
      get_yakl_instance().pool.finalize();
      get_yakl_instance().yakl_is_initialized = false;
      get_yakl_instance().pool_enabled = false;
      autotune::print_best();
      autotune::autotune_contexts.clear();
    } else {
      if constexpr (kokkos_debug) {
        std::cerr << "WARNING: Calling yakl::finalize() when YAKL is not initialized. ";
        std::cerr << "This might mean you've called yakl::finalize() more than once.\n";
      }
    }
    get_yakl_instance().yakl_is_initialized = false;
  }

}
