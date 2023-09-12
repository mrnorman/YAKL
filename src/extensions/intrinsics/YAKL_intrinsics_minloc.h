
#pragma once
// Included by YAKL_intrinsics.h

__YAKL_NAMESPACE_WRAPPER_BEGIN__
namespace yakl {
  namespace intrinsics {

    template <class T, int myStyle>
    inline int minloc( Array<T,1,memHost,myStyle> const &arr ) {
      #ifdef YAKL_DEBUG
        if (! allocated(arr)) yakl_throw("ERROR: calling minloc on an unallocated array");
      #endif
      T mv = minval(arr);
      if constexpr (myStyle == styleC) {
        for (int i=0; i < arr.totElems(); i++) { if (arr(i) == mv) return i; }
      } else {
        for (int i=lbound(arr,1); i <= ubound(arr,1); i++) { if (arr(i) == mv) return i; }
      }
      return -1;
    }

    template <class T>
    inline int minloc( Array<T,1,memDevice,styleC> const &arr , Stream stream = Stream()  ) {
      #ifdef YAKL_DEBUG
        if (! allocated(arr)) yakl_throw("ERROR: calling minloc on an unallocated array");
      #endif
      T mv = minval(arr,stream);
      #ifdef YAKL_B4B
        for (int i=0; i < arr.totElems(); i++) { if (arr(i) == mv) return i; }
        return -1;
      #else
        ScalarLiveOut<int> ind(0,stream);
        c::parallel_for( "YAKL_internal_minloc" , arr.totElems() , YAKL_LAMBDA (int i) { if (arr(i) == mv) ind = i; }, 
                         DefaultLaunchConfig().set_stream(stream) );
        return ind.hostRead(stream);
      #endif
    }

    template <class T>
    inline int minloc( Array<T,1,memDevice,styleFortran> const &arr , Stream stream = Stream()  ) {
      #ifdef YAKL_DEBUG
        if (! allocated(arr)) yakl_throw("ERROR: calling minloc on an unallocated array");
      #endif
      T mv = minval(arr,stream);
      #ifdef YAKL_B4B
        for (int i=lbound(arr,1); i <= ubound(arr,1); i++) { if (arr(i) == mv) return i; }
        return -1;
      #else
        ScalarLiveOut<int> ind(lbound(arr,1),stream);
        fortran::parallel_for( "YAKL_internal_minloc" , {lbound(arr,1),ubound(arr,1)} , YAKL_LAMBDA (int i) { if (arr(i) == mv) ind = i; }, 
                               DefaultLaunchConfig().set_stream(stream) );
        return ind.hostRead(stream);
      #endif
    }

    template <class T, class D0>
    YAKL_INLINE int minloc( FSArray<T,1,D0> const &arr ) {
      T m = arr.data()[0];
      int loc = lbound(arr,1);
      for (int i=lbound(arr,1); i<=ubound(arr,1); i++) {
        if (arr(i) < m) {
          m = arr(i);
          loc = i;
        }
      }
      return loc;
    }

    template <class T, index_t D0>
    YAKL_INLINE int minloc( SArray<T,1,D0> const &arr ) {
      T m = arr.data()[0];
      int loc = 0;
      for (int i=1; i < arr.get_dimensions()(0); i++) {
        if (arr(i) < m) {
          m = arr(i);
          loc = i;
        }
      }
      return loc;
    }

  }
}
__YAKL_NAMESPACE_WRAPPER_END__

