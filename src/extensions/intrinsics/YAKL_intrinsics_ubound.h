
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T> YAKL_INLINE int  ubound (T const &arr, int dim) { return arr.get_ubounds()(dim); }

    template <class T> YAKL_INLINE auto ubound (T const &arr) { return arr.get_ubounds(); }

  }
}

