
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <unsigned int n1, unsigned int n2, class real>
    YAKL_INLINE SArray<real,2,n2,n1> transpose(SArray<real,2,n1,n2> const &a) {
      SArray<real,2,n2,n1> ret;
      for (int j=0; j < n1; j++) {
        for (int i=0; i < n2; i++) {
          ret(j,i) = a(i,j);
        }
      }
      return ret;
    }

    template <int n1, int n2, class real>
    YAKL_INLINE FSArray<real,2,SB<n2>,SB<n1>> transpose(FSArray<real,2,SB<n1>,SB<n2>> const &a) {
      FSArray<real,2,SB<n2>,SB<n1>> ret;
      for (int j=1; j <= n1; j++) {
        for (int i=1; i <= n2; i++) {
          ret(j,i) = a(i,j);
        }
      }
      return ret;
    }

  }
}

