
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T1, class T2>
    YAKL_INLINE T1 sign(T1 a, T2 b) { return b >= 0 ? std::abs(a) : -std::abs(a); }

    template <class T1, class T2, int rank, int myStyle>
    inline Array<T1,rank,memHost,myStyle> sign( Array<T1,rank,memHost,myStyle> const & a ,
                                                Array<T2,rank,memHost,myStyle> const & b ) {
      #ifdef YAKL_DEBUG
        using yakl::componentwise::operator==;
        using yakl::componentwise::operator!;
        if (!allocated(a)) yakl_throw("ERROR: Calling sign with unallocated a");
        if (!allocated(b)) yakl_throw("ERROR: Calling sign with unallocated b");
        if (any(!(shape(a) == shape(b)))) yakl_throw("ERROR: Calling sign with differently arrays");
      #endif
      auto ret = a.createHostObject();
      for( int i=0; i < a.totElems(); i++) {
        ret.data()[i] = b.data()[i] >= 0 ? std::abs(a.data()[i]) : -std::abs(a.data()[i]);
      }
      return ret;
    }

    template <class T1, class T2, int rank, int myStyle>
    inline Array<T1,rank,memDevice,myStyle> sign( Array<T1,rank,memDevice,myStyle> const & a ,
                                                  Array<T2,rank,memDevice,myStyle> const & b , Stream stream = Stream() ) {
      #ifdef YAKL_DEBUG
        using yakl::componentwise::operator==;
        using yakl::componentwise::operator!;
        if (!allocated(a)) yakl_throw("ERROR: Calling sign with unallocated a");
        if (!allocated(b)) yakl_throw("ERROR: Calling sign with unallocated b");
        if (any(!(shape(a) == shape(b)))) yakl_throw("ERROR: Calling sign with differently arrays");
      #endif
      auto ret = a.createDeviceObject();
      parallel_for( "YAKL_internal_sign" , a.totElems() , YAKL_LAMBDA (int i) {
        ret.data()[i] = b.data()[i] >= 0 ? std::abs(a.data()[i]) : -std::abs(a.data()[i]);
      }, DefaultLaunchConfig().set_stream(stream) );
      ret.set_stream_dependency(stream);
      return ret;
    }

    template <class T1, class T2, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
    YAKL_INLINE SArray<T1,rank,D0,D1,D2,D3> sign( SArray<T1,rank,D0,D1,D2,D3> const & a ,
                                                  SArray<T2,rank,D0,D1,D2,D3> const & b ) {
      SArray<T1,rank,D0,D1,D2,D3> ret;
      for( int i=0; i < a.totElems(); i++) {
        ret.data()[i] = b.data()[i] >= 0 ? std::abs(a.data()[i]) : -std::abs(a.data()[i]);
      }
      return ret;
    }

    template <class T1, class T2, int rank, class B0, class B1, class B2, class B3>
    YAKL_INLINE FSArray<T1,rank,B0,B1,B2,B3> sign( FSArray<T1,rank,B0,B1,B2,B3> const & a ,
                                                   FSArray<T2,rank,B0,B1,B2,B3> const & b ) {
      FSArray<T1,rank,B0,B1,B2,B3> ret;
      for( int i=0; i < a.totElems(); i++) {
        ret.data()[i] = b.data()[i] >= 0 ? std::abs(a.data()[i]) : -std::abs(a.data()[i]);
      }
      return ret;
    }

  }
}

