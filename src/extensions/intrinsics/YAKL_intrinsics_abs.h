
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T, int rank, int myStyle>
    inline Array<T,rank,memHost,myStyle> abs( Array<T,rank,memHost,myStyle> const &arr ) {
      #ifdef KOKKOS_ENABLE_DEBUG
        if (!arr.initialized()) { Kokkos::abort("ERROR: calling abs on an array that has not been initialized"); }
      #endif
      auto ret = arr.createHostObject();
      for (int i=0; i < ret.totElems(); i++) { ret.data()[i] = std::abs(arr.data()[i]); };
      return ret;
    }

    template <class T, int rank, int myStyle>
    inline Array<T,rank,memDevice,myStyle> abs( Array<T,rank,memDevice,myStyle> const &arr ) {
      #ifdef KOKKOS_ENABLE_DEBUG
        if (!arr.initialized()) { Kokkos::abort("ERROR: calling abs on an array that has not been initialized"); }
      #endif
      auto ret = arr.createDeviceObject();
      c::parallel_for( YAKL_AUTO_LABEL() , ret.totElems() , KOKKOS_LAMBDA (int i) { ret.data()[i] = std::abs(arr.data()[i]); });
      return ret;
    }

    template <class T, int rank, size_t D0, size_t D1, size_t D2, size_t D3>
    KOKKOS_INLINE_FUNCTION SArray<T,rank,D0,D1,D2,D3> abs( SArray<T,rank,D0,D1,D2,D3> const &arr ) {
      SArray<T,rank,D0,D1,D2,D3> ret;
      for (int i=0; i < ret.totElems(); i++) { ret.data()[i] = std::abs(arr.data()[i]); };
      return ret;
    }

    template <class T, int rank, class B0, class B1, class B2, class B3>
    KOKKOS_INLINE_FUNCTION FSArray<T,rank,B0,B1,B2,B3> abs( FSArray<T,rank,B0,B1,B2,B3> const &arr ) {
      FSArray<T,rank,B0,B1,B2,B3> ret;
      for (int i=0; i < ret.totElems(); i++) { ret.data()[i] = std::abs(arr.data()[i]); };
      return ret;
    }

  }
}

