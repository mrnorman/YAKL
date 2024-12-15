
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T> KOKKOS_INLINE_FUNCTION int  size(T const &arr, int dim) {
      #ifdef KOKKOS_ENABLE_DEBUG
        if (!allocated(arr)) Kokkos::abort("ERROR: Calling size on an unallocated array");
      #endif
      return arr.get_dimensions()(dim);
    }

    template <class T> KOKKOS_INLINE_FUNCTION int  size(T const &arr) {
      #ifdef KOKKOS_ENABLE_DEBUG
        if (!allocated(arr)) Kokkos::abort("ERROR: Calling size on an unallocated array");
      #endif
      return arr.totElems();
    }

  }
}

