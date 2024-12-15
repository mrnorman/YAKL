
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T> KOKKOS_INLINE_FUNCTION auto shape(T const &arr) {
      #ifdef KOKKOS_ENABLE_DEBUG
        if (!allocated(arr)) Kokkos::abort("ERROR: Calling shape on an unallocated array");
      #endif
      return arr.get_dimensions();
    }

  }
}

