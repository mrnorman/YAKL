
#pragma once
// Included by YAKL_intrinsics.h

__YAKL_NAMESPACE_WRAPPER_BEGIN__
namespace yakl {
  namespace intrinsics {

    template <int rank, int myStyle>
    inline int count( Array<bool,rank,memHost,myStyle> const &mask ) {
      #ifdef YAKL_DEBUG
        if (!mask.initialized()) { yakl_throw("ERROR: calling count on an array that has not been initialized"); }
      #endif
      int numTrue = 0;
      for (int i=0; i < mask.totElems(); i++) {
        if (mask.data()[i]) { numTrue++; }
      }
      return numTrue;
    }

    template <int rank, int myStyle>
    inline int count( Array<bool,rank,memDevice,myStyle> const &mask , Stream stream = Stream() ) {
      #ifdef YAKL_DEBUG
        if (!mask.initialized()) { yakl_throw("ERROR: calling count on an array that has not been initialized"); }
      #endif
      auto intarr = mask.template createDeviceObject<int>();
      c::parallel_for( "YAKL_internal_count" , mask.totElems() , YAKL_LAMBDA (int i) { intarr.data()[i] = mask.data()[i] ? 1 : 0; },
                       DefaultLaunchConfig().set_stream(stream) );
      return yakl::intrinsics::sum(intarr,stream);
    }

    template <int rank, class D0, class D1, class D2, class D3>
    YAKL_INLINE int count( FSArray<bool,rank,D0,D1,D2,D3> const &mask ) {
      int numTrue = 0;
      for (int i=0; i < mask.totElems(); i++) {
        if (mask.data()[i]) { numTrue++; }
      }
      return numTrue;
    }

    template <int rank, index_t D0, index_t D1, index_t D2, index_t D3>
    YAKL_INLINE int count( SArray<bool,rank,D0,D1,D2,D3> const &mask ) {
      int numTrue = 0;
      for (int i=0; i < mask.totElems(); i++) {
        if (mask.data()[i]) { numTrue++; }
      }
      return numTrue;
    }

  }
}
__YAKL_NAMESPACE_WRAPPER_END__

