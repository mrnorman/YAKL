
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T, int myStyle>
    inline int maxloc( Array<T,1,memHost,myStyle> const &arr ) {
      #ifdef KOKKOS_ENABLE_DEBUG
        if (! allocated(arr)) Kokkos::abort("ERROR: calling maxloc on an unallocated array");
      #endif
      T mv = maxval(arr);
      if constexpr (myStyle == styleC) {
        for (int i=0; i < arr.totElems(); i++) { if (arr(i) == mv) return i; }
      } else {
        for (int i=lbound(arr,1); i <= ubound(arr,1); i++) { if (arr(i) == mv) return i; }
      }
      return -1;
    }

    template <class T>
    inline int maxloc( Array<T,1,memDevice,styleC> const &arr ) {
      #ifdef KOKKOS_ENABLE_DEBUG
        if (! allocated(arr)) Kokkos::abort("ERROR: calling maxloc on an unallocated array");
      #endif
      T mv = maxval(arr);
      #ifdef YAKL_B4B
        for (int i=0; i < arr.totElems(); i++) { if (arr(i) == mv) return i; }
        return -1;
      #else
        ScalarLiveOut<int> ind(0);
        c::parallel_for( YAKL_AUTO_LABEL() , arr.totElems() , KOKKOS_LAMBDA (int i) { if (arr(i) == mv) ind = i; });
        return ind.hostRead();
      #endif
    }

    template <class T>
    inline int maxloc( Array<T,1,memDevice,styleFortran> const &arr ) {
      #ifdef KOKKOS_ENABLE_DEBUG
        if (! allocated(arr)) Kokkos::abort("ERROR: calling maxloc on an unallocated array");
      #endif
      T mv = maxval(arr);
      #ifdef YAKL_B4B
        for (int i=lbound(arr,1); i <= ubound(arr,1); i++) { if (arr(i) == mv) return i; }
        return -1;
      #else
        ScalarLiveOut<int> ind(lbound(arr,1));
        fortran::parallel_for( YAKL_AUTO_LABEL() , {lbound(arr,1),ubound(arr,1)} , KOKKOS_LAMBDA (int i) { if (arr(i) == mv) ind = i; });
        return ind.hostRead();
      #endif
    }

    template <class T, class D0>
    KOKKOS_INLINE_FUNCTION int maxloc( FSArray<T,1,D0> const &arr ) {
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

    template <class T, size_t D0>
    KOKKOS_INLINE_FUNCTION int maxloc( SArray<T,1,D0> const &arr ) {
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

