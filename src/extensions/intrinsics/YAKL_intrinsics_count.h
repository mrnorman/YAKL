
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <int rank, int myStyle>
    inline int count( Array<bool,rank,memHost,myStyle> const &mask ) {
      #ifdef KOKKOS_ENABLE_DEBUG
        if (!mask.initialized()) { Kokkos::abort("ERROR: calling count on an array that has not been initialized"); }
      #endif
      int numTrue = 0;
      for (int i=0; i < mask.totElems(); i++) {
        if (mask.data()[i]) { numTrue++; }
      }
      return numTrue;
    }

    template <int rank, int myStyle>
    inline int count( Array<bool,rank,memDevice,myStyle> const &mask ) {
      #ifdef KOKKOS_ENABLE_DEBUG
        if (!mask.initialized()) { Kokkos::abort("ERROR: calling count on an array that has not been initialized"); }
      #endif
      auto intarr = mask.template createDeviceObject<int>();
      c::parallel_for( YAKL_AUTO_LABEL() , mask.totElems() , KOKKOS_LAMBDA (int i) { intarr.data()[i] = mask.data()[i] ? 1 : 0; });
      return yakl::intrinsics::sum(intarr);
    }

    template <int rank, class D0, class D1, class D2, class D3>
    KOKKOS_INLINE_FUNCTION int count( FSArray<bool,rank,D0,D1,D2,D3> const &mask ) {
      int numTrue = 0;
      for (int i=0; i < mask.totElems(); i++) {
        if (mask.data()[i]) { numTrue++; }
      }
      return numTrue;
    }

    template <int rank, size_t D0, size_t D1, size_t D2, size_t D3>
    KOKKOS_INLINE_FUNCTION int count( SArray<bool,rank,D0,D1,D2,D3> const &mask ) {
      int numTrue = 0;
      for (int i=0; i < mask.totElems(); i++) {
        if (mask.data()[i]) { numTrue++; }
      }
      return numTrue;
    }

  }
}

