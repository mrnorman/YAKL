
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T, int myMem, int myStyle>
    inline Array<T,2,myMem,myStyle> transpose(Array<T,2,myMem,myStyle> const &in, Stream stream = Stream() ) {
      #ifdef YAKL_DEBUG
        if (!allocated(in)) yakl_throw("ERROR: Calling transpose on unallocated array");
      #endif
      if constexpr (myStyle == styleC) {
        auto d0 = size(in,0);
        auto d1 = size(in,1);
        if constexpr (myMem == memHost) {
          auto out = in.createHostCopy().template reshape<2>( { d1 , d0 } );
          for (int i=0; i < d0; i++) {
            for (int j=0; j < d1; j++) {
              out(j,i) = in(i,j);
            }
          }
          return out;
        } else {
          auto out = in.createDeviceCopy(stream).template reshape<2>( { d1 , d0 } );
          c::parallel_for( "YAKL_internal_transpose" , c::Bounds<2>(d0,d1) , YAKL_LAMBDA (int i, int j) {
            out(j,i) = in(i,j);
          } , DefaultLaunchConfig().set_stream(stream) );
          out.add_stream_dependency(stream);
          return out;
        }
      } else {
        auto l1 = lbound(in,1);
        auto l2 = lbound(in,2);
        auto u1 = ubound(in,1);
        auto u2 = ubound(in,2);
        if constexpr (myMem == memHost) {
          auto out = in.createHostCopy().template reshape<2>( { {l2,u2} , {l1,u1} } );
          for (int i=l1; i <= u1; i++) {
            for (int j=l2; j <= u2; j++) {
              out(j,i) = in(i,j);
            }
          }
          return out;
        } else {
          auto out = in.createDeviceCopy(stream).template reshape<2>( { {l2,u2} , {l1,u1} } );
          fortran::parallel_for( "YAKL_internal_transpose" , fortran::Bounds<2>({l1,u1},{l2,u2}) , YAKL_LAMBDA (int i, int j) {
            out(j,i) = in(i,j);
          } , DefaultLaunchConfig().set_stream(stream) );
          out.add_stream_dependency(stream);
          return out;
        }
      }
      // Can't get here, but nvcc isn't smart enough to know that evidently.
      return Array<T,2,myMem,myStyle>();
    }

    template <unsigned int n1, unsigned int n2, class T>
    YAKL_INLINE SArray<T,2,n2,n1> transpose(SArray<T,2,n1,n2> const &a) {
      SArray<T,2,n2,n1> ret;
      for (int j=0; j < n1; j++) {
        for (int i=0; i < n2; i++) {
          ret(j,i) = a(i,j);
        }
      }
      return ret;
    }

    template <class B1, class B2, class T>
    YAKL_INLINE FSArray<T,2,B1,B2> transpose(FSArray<T,2,B1,B2> const &a) {
      FSArray<T,2,B2,B1> ret;
      for (int j=B1::lower(); j <= B1::upper(); j++) {
        for (int i=B2::lower(); i <= B2::upper(); i++) {
          ret(j,i) = a(i,j);
        }
      }
      return ret;
    }

  }
}

