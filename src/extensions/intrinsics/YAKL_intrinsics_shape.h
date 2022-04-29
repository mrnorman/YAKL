
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T> YAKL_INLINE auto shape(T const &arr) {
      #ifdef YAKL_DEBUG
        if (!allocated(arr)) yakl_throw("ERROR: Calling shape on an unallocated array");
      #endif
      return arr.get_dimensions();
    }

  }
}

