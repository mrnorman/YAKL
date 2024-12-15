
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T, int rank, int myStyle>
    inline T product( Array<T,rank,memHost,myStyle> const &arr ) {
      #ifdef KOKKOS_ENABLE_DEBUG
        if (!arr.initialized()) { Kokkos::abort("ERROR: calling product on an array that has not been initialized"); }
      #endif
      typename std::remove_cv<T>::type m = arr.data()[0];
      for (int i=1; i<arr.totElems(); i++) { m *= arr.data()[i]; }
      return m;
    }

    template <class T, int rank, int myStyle>
    inline T product( Array<T,rank,memDevice,myStyle> const &arr ) {
      #ifdef KOKKOS_ENABLE_DEBUG
        if (!arr.initialized()) { Kokkos::abort("ERROR: calling product on an array that has not been initialized"); }
      #endif
      typedef typename std::remove_cv<T>::type TNC;  // T Non-Const
      TNC result;
      Kokkos::parallel_reduce( YAKL_AUTO_LABEL() , arr.size() , KOKKOS_LAMBDA (int i, TNC & lprod) {
        lprod *= arr.data()[i];
      }, Kokkos::Prod<TNC>(result) );
      return result;
    }

    template <class T, int rank, class D0, class D1, class D2, class D3>
    KOKKOS_INLINE_FUNCTION T product( FSArray<T,rank,D0,D1,D2,D3> const &arr ) {
      T m = arr.data()[0];
      for (int i=1; i<arr.totElems(); i++) { m *= arr.data()[i]; }
      return m;
    }

    template <class T, int rank, size_t D0, size_t D1, size_t D2, size_t D3>
    KOKKOS_INLINE_FUNCTION T product( SArray<T,rank,D0,D1,D2,D3> const &arr ) {
      T m = arr.data()[0];
      for (int i=1; i<arr.totElems(); i++) { m *= arr.data()[i]; }
      return m;
    }

  }
}

