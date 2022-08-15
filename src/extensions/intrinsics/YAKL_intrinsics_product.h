
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T, int rank, int myStyle>
    inline T product( Array<T,rank,memHost,myStyle> const &arr ) {
      #ifdef YAKL_DEBUG
        if (!arr.initialized()) { yakl_throw("ERROR: calling product on an array that has not been initialized"); }
      #endif
      typename std::remove_cv<T>::type m = arr.data()[0];
      for (int i=1; i<arr.totElems(); i++) { m *= arr.data()[i]; }
      return m;
    }

    template <class T, int rank, int myStyle>
    inline T product( Array<T,rank,memDevice,myStyle> const &arr , Stream stream = Stream() ) {
      #ifdef YAKL_DEBUG
        if (!arr.initialized()) { yakl_throw("ERROR: calling product on an array that has not been initialized"); }
      #endif
      typedef typename std::remove_cv<T>::type TNC;  // T Non-Const
      ParallelProd<TNC,memDevice> pprod(arr.totElems(),stream);
      return pprod( const_cast<TNC *>(arr.data()) );
    }

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

