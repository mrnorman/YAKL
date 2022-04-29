
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T, class D0>
    YAKL_INLINE int maxloc( FSArray<T,1,D0> const &arr ) {
      T m = arr.data()[0];
      int loc = lbound(arr,1);
      for (int i=lbound(arr,1); i<=ubound(arr,1); i++) {
        if (arr(i) > m) {
          m = arr(i);
          loc = i;
        }
      }
      return loc;
    }
    template <class T, unsigned D0>
    YAKL_INLINE int maxloc( SArray<T,1,D0> const &arr ) {
      T m = arr.data()[0];
      int loc = 0;
      for (int i=1; i<arr.get_dimensions()(0); i++) {
        if (arr(i) > m) {
          m = arr(i);
          loc = i;
        }
      }
      return loc;
    }

  }
}

