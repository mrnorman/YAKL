
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T> YAKL_INLINE int  ubound (T const &arr, int dim) {
      #ifdef YAKL_DEBUG
        if (!allocated(arr)) yakl_throw("ERROR: Calling ubound on an unallocated array");
      #endif
      return arr.get_ubounds()(dim);
    }

    template <class T> YAKL_INLINE auto ubound (T const &arr) {
      #ifdef YAKL_DEBUG
        if (!allocated(arr)) yakl_throw("ERROR: Calling ubound on an unallocated array");
      #endif
      return arr.get_ubounds();
    }

  }
}

