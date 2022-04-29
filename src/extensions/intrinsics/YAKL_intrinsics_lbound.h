
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T> YAKL_INLINE int  lbound (T const &arr, int dim) { return arr.get_lbounds()(dim); }

    template <class T> YAKL_INLINE auto lbound (T const &arr) { return arr.get_lbounds(); }

  }
}

