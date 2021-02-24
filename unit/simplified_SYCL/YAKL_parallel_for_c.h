
#pragma once

namespace c {

  template<> class Bounds<1,true> {
  public:
    index_t nIter;
    index_t dims[1];
    Bounds( index_t b0 ) {
      dims[0] = b0;
      nIter = dims[0];
    }
    YAKL_INLINE void unpackIndices( index_t iGlob , int indices[1] ) const {
      // Compute base indices
      indices[0] = iGlob;
    }
  };

  template <int N> using SimpleBounds = Bounds<N,true>;

  template <class F, bool simple> YAKL_INLINE void callFunctor(F const &f , Bounds<1,simple> const &bnd , int const i ) {
    int ind[1];
    bnd.unpackIndices( i , ind );
    f(ind[0]);
  }

  template<class F, int N, bool simple>
  void parallel_for_sycl( Bounds<N,simple> const &bounds , F const &f , int vectorSize = 128 ) {
    sycl_default_stream.parallel_for<class sycl_kernel>( sycl::range<1>(bounds.nIter) , [=] (sycl::id<1> i) {
      callFunctor( f , bounds , i );
    });
    check_last_error();
  }

  template <class F, int N, bool simple>
  inline void parallel_for( Bounds<N,simple> const &bounds , F const &f , int vectorSize = 128 ) {
      parallel_for_sycl( bounds , f , vectorSize );
  }


}
