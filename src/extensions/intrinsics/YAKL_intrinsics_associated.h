
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T> KOKKOS_INLINE_FUNCTION bool associated (T const &arr) { return arr.data() != nullptr; }

  }
}

