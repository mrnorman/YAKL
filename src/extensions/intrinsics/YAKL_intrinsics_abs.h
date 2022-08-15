
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T, int rank, int myStyle>
    inline Array<T,rank,memHost,myStyle> abs( Array<T,rank,memHost,myStyle> const &arr ) {
      #ifdef YAKL_DEBUG
        if (!arr.initialized()) { yakl_throw("ERROR: calling abs on an array that has not been initialized"); }
      #endif
      auto ret = arr.createHostObject();
      for (int i=0; i < ret.totElems(); i++) { ret.data()[i] = std::abs(arr.data()[i]); };
      return ret;
    }

    template <class T, int rank, int myStyle>
    inline Array<T,rank,memDevice,myStyle> abs( Array<T,rank,memDevice,myStyle> const &arr , Stream stream = Stream() ) {
      #ifdef YAKL_DEBUG
        if (!arr.initialized()) { yakl_throw("ERROR: calling abs on an array that has not been initialized"); }
      #endif
      auto ret = arr.createDeviceObject();
      c::parallel_for( "YAKL_internal_abs" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = std::abs(arr.data()[i]); },
                       DefaultLaunchConfig().set_stream(stream) );
      ret.add_stream_dependency(stream);
      return ret;
    }

    template <class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
    YAKL_INLINE SArray<T,rank,D0,D1,D2,D3> abs( SArray<T,rank,D0,D1,D2,D3> const &arr ) {
      SArray<T,rank,D0,D1,D2,D3> ret;
      for (int i=0; i < ret.totElems(); i++) { ret.data()[i] = std::abs(arr.data()[i]); };
      return ret;
    }

    template <class T, int rank, class B0, class B1, class B2, class B3>
    YAKL_INLINE FSArray<T,rank,B0,B1,B2,B3> abs( FSArray<T,rank,B0,B1,B2,B3> const &arr ) {
      FSArray<T,rank,B0,B1,B2,B3> ret;
      for (int i=0; i < ret.totElems(); i++) { ret.data()[i] = std::abs(arr.data()[i]); };
      return ret;
    }

  }
}

