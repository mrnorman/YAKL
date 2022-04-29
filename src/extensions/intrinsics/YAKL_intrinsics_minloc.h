
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T, int myMem, int myStyle>
    inline int minloc( Array<T,1,myMem,myStyle> const &arr ) {
      T mv = minval(arr);
      if constexpr (myMem == memHost) {
        if constexpr (myStyle == styleC) {
          for (int i=0; i < arr.totElems(); i++) { if (arr(i) == mv) return i; }
        } else {
          for (int i=lbound(arr,1); i <= ubound(arr,1); i++) { if (arr(i) == mv) return i; }
        }
      } else {
        #ifdef YAKL_B4B
          if constexpr (myStyle == styleC) {
            for (int i=0; i < arr.totElems(); i++) { if (arr(i) == mv) return i; }
          } else {
            for (int i=lbound(arr,1); i <= ubound(arr,1); i++) { if (arr(i) == mv) return i; }
          }
        #else
          if constexpr (myStyle == styleC) {
            ScalarLiveOut<int> ind(0);
            c::parallel_for( arr.totElems() , YAKL_LAMBDA (int i) { if (arr(i) == mv) ind = i; });
            return ind.hostRead();
          } else {
            ScalarLiveOut<int> ind(lbound(arr,1));
            fortran::parallel_for( {lbound(arr,1),ubound(arr,1)} , YAKL_LAMBDA (int i) { if (arr(i) == mv) ind = i; });
            return ind.hostRead();
          }
        #endif
      }
      // Never reaches here, but nvcc isn't smart enough to figure it out.
      return 0;
    }

    template <class T, class D0>
    YAKL_INLINE int minloc( FSArray<T,1,D0> const &arr ) {
      T m = arr.data()[0];
      int loc = lbound(arr,1);
      for (int i=lbound(arr,1); i<=ubound(arr,1); i++) {
        if (arr(i) < m) {
          m = arr(i);
          loc = i;
        }
      }
      return loc;
    }

    template <class T, unsigned D0>
    YAKL_INLINE int minloc( SArray<T,1,D0> const &arr ) {
      T m = arr.data()[0];
      int loc = 0;
      for (int i=1; i < arr.get_dimensions()(0); i++) {
        if (arr(i) < m) {
          m = arr(i);
          loc = i;
        }
      }
      return loc;
    }

  }
}

