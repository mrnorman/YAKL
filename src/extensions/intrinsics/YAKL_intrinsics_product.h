
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T, int rank, class D0, class D1, class D2, class D3>
    YAKL_INLINE T product( FSArray<T,rank,D0,D1,D2,D3> const &arr ) {
      T m = arr.data()[0];
      for (int i=1; i<arr.totElems(); i++) { m *= arr.data()[i]; }
      return m;
    }

    template <class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
    YAKL_INLINE T product( SArray<T,rank,D0,D1,D2,D3> const &arr ) {
      T m = arr.data()[0];
      for (int i=1; i<arr.totElems(); i++) { m *= arr.data()[i]; }
      return m;
    }

  }
}

