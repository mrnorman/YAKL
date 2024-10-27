
#pragma once
// Included by YAKL_intrinsics.h

__YAKL_NAMESPACE_WRAPPER_BEGIN__
namespace yakl {
  namespace intrinsics {

    template <class T, int rank, int myStyle>
    inline T sum( Array<T,rank,memHost,myStyle> const &arr ) {
      #ifdef YAKL_DEBUG
        if (!arr.initialized()) { yakl_throw("ERROR: calling sum on an array that has not been initialized"); }
      #endif
      typename std::remove_cv<T>::type m = arr.data()[0];
      for (int i=1; i<arr.totElems(); i++) { m += arr.data()[i]; }
      return m;
    }

    template <class T, int rank, int myStyle>
    inline T sum( Array<T,rank,memDevice,myStyle> const &arr , Stream stream = Stream() ) {
      #ifdef YAKL_DEBUG
        if (!arr.initialized()) { yakl_throw("ERROR: calling sum on an array that has not been initialized"); }
      #endif
      typedef typename std::remove_cv<T>::type TNC;  // T Non-Const
      ParallelSum<TNC,memDevice> psum(arr.totElems(),stream);
      return psum( const_cast<TNC *>(arr.data()) );
    }

    template <class T, int rank, class D0, class D1, class D2, class D3>
    YAKL_INLINE T sum( FSArray<T,rank,D0,D1,D2,D3> const &arr ) {
      typename std::remove_cv<T>::type m = arr.data()[0];
      for (int i=1; i<arr.totElems(); i++) { m += arr.data()[i]; }
      return m;
    }

    template <class T, int rank, index_t D0, index_t D1, index_t D2, index_t D3>
    YAKL_INLINE T sum( SArray<T,rank,D0,D1,D2,D3> const &arr ) {
      typename std::remove_cv<T>::type m = arr.data()[0];
      for (int i=1; i<arr.totElems(); i++) { m += arr.data()[i]; }
      return m;
    }

  }
}
__YAKL_NAMESPACE_WRAPPER_END__

