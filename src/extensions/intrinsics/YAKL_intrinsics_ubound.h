
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T> KOKKOS_INLINE_FUNCTION int  ubound (T const &arr, int dim) {
      #ifdef KOKKOS_ENABLE_DEBUG
        if (!allocated(arr)) Kokkos::abort("ERROR: Calling ubound on an unallocated array");
      #endif
      return arr.get_ubounds()(dim);
    }

    template <class T> KOKKOS_INLINE_FUNCTION auto ubound (T const &arr) {
      #ifdef KOKKOS_ENABLE_DEBUG
        if (!allocated(arr)) Kokkos::abort("ERROR: Calling ubound on an unallocated array");
      #endif
      return arr.get_ubounds();
    }

  }
}

