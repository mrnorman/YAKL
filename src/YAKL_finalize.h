
#pragma once
// Included by YAKL.h

namespace yakl {

  inline void finalize() {
    // Only finalize if YAKL's already initialized
    if ( get_yakl_instance().is_initialized() ) {
      Kokkos::fence();  // Make sure all device work is done before we start freeing pool memory
      get_yakl_instance().pool.finalize();
      get_yakl_instance().yakl_is_initialized = false;
      get_yakl_instance().pool_enabled = false;
    } else {
      if constexpr (kokkos_debug) {
        std::cerr << "WARNING: Calling yakl::finalize() when YAKL is not initialized. ";
        std::cerr << "This might mean you've called yakl::finalize() more than once.\n";
      }
    }
    get_yakl_instance().yakl_is_initialized = false;
  }

}


