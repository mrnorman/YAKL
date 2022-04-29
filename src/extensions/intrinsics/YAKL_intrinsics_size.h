
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T> YAKL_INLINE int  size(T const &arr, int dim) { return arr.get_dimensions()(dim); }

    template <class T> YAKL_INLINE int  size(T const &arr) { return arr.totElems(); }

  }
}

