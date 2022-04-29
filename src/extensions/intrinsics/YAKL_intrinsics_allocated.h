
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T> YAKL_INLINE bool allocated (T const &arr) { return arr.data() != nullptr; }

  }
}

