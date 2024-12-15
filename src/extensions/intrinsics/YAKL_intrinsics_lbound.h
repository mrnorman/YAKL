
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T> KOKKOS_INLINE_FUNCTION int  lbound (T const &arr, int dim) {
      #ifdef KOKKOS_ENABLE_DEBUG
        if ( ! allocated( arr ) ) Kokkos::abort("ERROR: calling lbound on an unallocated array");
      #endif
      return arr.get_lbounds()(dim);
    }

    template <class T> KOKKOS_INLINE_FUNCTION auto lbound (T const &arr) { 
      #ifdef KOKKOS_ENABLE_DEBUG
        if ( ! allocated( arr ) ) Kokkos::abort("ERROR: calling lbound on an unallocated array");
      #endif
      return arr.get_lbounds();
    }

  }
}

