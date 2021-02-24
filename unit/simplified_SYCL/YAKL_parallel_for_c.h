
#pragma once

namespace c {

  class LBnd {
  public:
    int l, u, s;
    LBnd(int u) {
      this->l = 0;
      this->u = u-1;
      this->s = 1;
    }
    LBnd(int l, int u) {
      this->l = l;
      this->u = u;
      this->s = 1;
      //if (u < l) yakl_throw("ERROR: cannot specify an upper bound < lower bound");
    }
    LBnd(int l, int u, int s) {
      this->l = l;
      this->u = u;
      this->s = s;
      //if (s < 1) yakl_throw("ERROR: negative strides not yet supported.");
    }
  };

  template <int N, bool simple = false> class Bounds;

  template<> class Bounds<1,false> {
  public:
    index_t nIter;
    int     lbounds[1];
    index_t dims[1];
    index_t strides[1];
    Bounds( LBnd const &b0 ) {
      lbounds[0] = b0.l;   strides[0] =  b0.s;   dims[0] = ( b0.u - b0.l + 1 ) / b0.s;
      nIter = dims[0];
    }
    YAKL_INLINE void unpackIndices( index_t iGlob , int indices[1] ) const {
      // Compute base indices
      indices[0] = iGlob;
      // Apply strides and lower bounds
      indices[0] = indices[0]*strides[0] + lbounds[0];
    }
  };

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
  }

  template <class F, int N, bool simple>
  inline void parallel_for( Bounds<N,simple> const &bounds , F const &f , int vectorSize = 128 ) {
      parallel_for_sycl( bounds , f , vectorSize );
  }


}
