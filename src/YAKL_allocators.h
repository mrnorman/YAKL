
#pragma once
// Included by YAKL.h

namespace yakl {

  inline void * alloc_device( size_t bytes, char const *label) {
    if (get_yakl_instance().use_pool()) { return get_yakl_instance().pool.allocate(bytes,label); }
    else                                { return Kokkos::kokkos_malloc( label , bytes ); }
  }


  inline void   free_device ( void * ptr  , char const *label) {
    if (get_yakl_instance().use_pool()) { get_yakl_instance().pool.free(ptr,label); }
    else                                { Kokkos::kokkos_free(ptr);                 }
  }

}


