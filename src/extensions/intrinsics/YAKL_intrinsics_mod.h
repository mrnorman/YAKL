
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T1, class T2>
    YAKL_INLINE decltype(T1() - ((int)(T1()/T2()) * T2())) mod(T1 a, T2 b) { return a - ((int)(a/b) * b); }

  }
}

