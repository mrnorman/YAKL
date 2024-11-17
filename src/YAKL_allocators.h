
#pragma once
// Included by YAKL.h

namespace yakl {

  /**
   * @brief If true, then the pool allocator is being used for all device allocations
   */
  inline bool use_pool() { return get_yakl_instance().pool_enabled; }


  inline LinearAllocator & get_pool() { return get_yakl_instance().pool; }


  /** @brief Allocate on the device using YAKL's device allocator */
  inline void * alloc_device( size_t bytes, char const *label) {
    if (use_pool()) { return get_yakl_instance().pool.allocate(bytes,label); }
    else            {
      #ifdef YAKL_MANAGED_MEMORY
        return Kokkos::kokkos_malloc<Kokkos::SharedSpace>( label , bytes );
      #else
        return Kokkos::kokkos_malloc( label , bytes );
      #endif
    }
  }


  /** @brief Free on the device using YAKL's device deallocator */
  inline void   free_device ( void * ptr  , char const *label) {
    if (use_pool()) { get_yakl_instance().pool.free(ptr,label); }
    else            { Kokkos::kokkos_free(ptr);                 }
  }

}


