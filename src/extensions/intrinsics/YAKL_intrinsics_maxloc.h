
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T, int myStyle>
    inline int maxloc( Array<T,1,memHost,myStyle> const &arr ) {
      #ifdef YAKL_DEBUG
        if (! allocated(arr)) yakl_throw("ERROR: calling maxloc on an unallocated array");
      #endif
      T mv = maxval(arr);
      if constexpr (myStyle == styleC) {
        for (int i=0; i < arr.totElems(); i++) { if (arr(i) == mv) return i; }
      } else {
        for (int i=lbound(arr,1); i <= ubound(arr,1); i++) { if (arr(i) == mv) return i; }
      }
      // Never reaches here, but nvcc isn't smart enough to figure it out.
      return 0;
    }

    template <class T>
    inline int maxloc( Array<T,1,memDevice,styleC> const &arr , Stream stream = Stream()  ) {
      #ifdef YAKL_DEBUG
        if (! allocated(arr)) yakl_throw("ERROR: calling maxloc on an unallocated array");
      #endif
      T mv = maxval(arr,stream);
      #ifdef YAKL_B4B
        for (int i=0; i < arr.totElems(); i++) { if (arr(i) == mv) return i; }
      #else
        ScalarLiveOut<int> ind(0,stream);
        c::parallel_for( "YAKL_internal_maxloc" , arr.totElems() , YAKL_LAMBDA (int i) { if (arr(i) == mv) ind = i; }, 
                         DefaultLaunchConfig().set_stream(stream) );
        return ind.hostRead(stream);
      #endif
      // Never reaches here, but nvcc isn't smart enough to figure it out.
      return 0;
    }

    template <class T>
    inline int maxloc( Array<T,1,memDevice,styleFortran> const &arr , Stream stream = Stream()  ) {
      #ifdef YAKL_DEBUG
        if (! allocated(arr)) yakl_throw("ERROR: calling maxloc on an unallocated array");
      #endif
      T mv = maxval(arr,stream);
      #ifdef YAKL_B4B
        for (int i=lbound(arr,1); i <= ubound(arr,1); i++) { if (arr(i) == mv) return i; }
      #else
        ScalarLiveOut<int> ind(lbound(arr,1),stream);
        fortran::parallel_for( "YAKL_internal_maxloc" , {lbound(arr,1),ubound(arr,1)} , YAKL_LAMBDA (int i) { if (arr(i) == mv) ind = i; }, 
                               DefaultLaunchConfig().set_stream(stream) );
        return ind.hostRead(stream);
      #endif
      // Never reaches here, but nvcc isn't smart enough to figure it out.
      return 0;
    }

    template <class T, class D0>
    YAKL_INLINE int maxloc( FSArray<T,1,D0> const &arr ) {
      T m = arr.data()[0];
      int loc = lbound(arr,1);
      for (int i=lbound(arr,1); i<=ubound(arr,1); i++) {
        if (arr(i) > m) {
          m = arr(i);
          loc = i;
        }
      }
      return loc;
    }

    template <class T, unsigned D0>
    YAKL_INLINE int maxloc( SArray<T,1,D0> const &arr ) {
      T m = arr.data()[0];
      int loc = 0;
      for (int i=1; i<arr.get_dimensions()(0); i++) {
        if (arr(i) > m) {
          m = arr(i);
          loc = i;
        }
      }
      return loc;
    }

  }
}

