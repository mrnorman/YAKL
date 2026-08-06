
#pragma once
// Included by YAKL.h

namespace yakl {

  inline void * alloc_device( size_t bytes, char const *label) {
    if (!get_yakl_instance().is_initialized()) {
      Kokkos::abort("ERROR: alloc_device called outside yakl::init() / yakl::finalize()");
    }
    if constexpr (kokkos_debug) {
      if (!Kokkos::is_initialized()) Kokkos::abort("ERROR: alloc_device called before Kokkos initialization");
      if (label == nullptr) Kokkos::abort("ERROR: alloc_device received a null label");
    }
    void * ptr;
    if (get_yakl_instance().use_pool()) { ptr = get_yakl_instance().pool.allocate(bytes,label); }
    else                                { ptr = Kokkos::kokkos_malloc( label , bytes ); }
    if (ptr != nullptr) get_yakl_instance().num_device_allocations++;
    return ptr;
  }


  inline void   free_device ( void * ptr  , char const *label) {
    if (!get_yakl_instance().is_initialized()) {
      Kokkos::abort("ERROR: free_device called outside yakl::init() / yakl::finalize()");
    }
    if (get_yakl_instance().num_device_allocations == 0) {
      Kokkos::abort("ERROR: free_device called with no live YAKL device allocations");
    }
    if constexpr (kokkos_debug) {
      if (!Kokkos::is_initialized()) Kokkos::abort("ERROR: free_device called before Kokkos initialization");
      if (ptr == nullptr) Kokkos::abort("ERROR: free_device received a null pointer");
      if (label == nullptr) Kokkos::abort("ERROR: free_device received a null label");
    }
    if (get_yakl_instance().use_pool()) { get_yakl_instance().pool.free(ptr,label); }
    else                                { Kokkos::kokkos_free(ptr);                 }
    get_yakl_instance().num_device_allocations--;
  }

}
