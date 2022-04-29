
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T> YAKL_INLINE auto shape(T const &arr) { return arr.get_dimensions(); }

  }
}

