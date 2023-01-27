/**
 * @file
 * YAKL routines to set entire arrays to a given value. 
 * [Deprecated for user use]: Please use operator= instead.
 */

#pragma once
// Included by YAKL.h

namespace yakl {

  /** @private */
  template <class T, int rank, int myMem, int myStyle, class I>
  void memset( Array<T,rank,myMem,myStyle> &arr , I val , Stream stream = Stream() ) {
    #ifdef YAKL_DEBUG
      if (! arr.initialized()) {
        yakl_throw("ERROR: calling memset on an array that is not allocated");
      }
    #endif

    // Use memset for zero values when possible
    if (myMem == memDevice && val == 0) {
      #if   defined(YAKL_ARCH_CUDA)
        cudaMemsetAsync( arr.data() , 0 , sizeof(T)*arr.totElems() , stream.get_real_stream() );
      #elif defined(YAKL_ARCH_HIP)
        hipMemsetAsync ( arr.data() , 0 , sizeof(T)*arr.totElems() , stream.get_real_stream() );
      #elif defined(YAKL_ARCH_SYCL)
        stream.get_real_stream().memset( arr.data() , 0 , sizeof(T)*arr.totElems() );
      #else
        c::parallel_for( "YAKL_internal_memset" , arr.totElems() , YAKL_LAMBDA (int i) {
          arr.data()[i] = 0;
        } , DefaultLaunchConfig().set_stream(stream) );
      #endif
    } else {
      // SYCL has a fill routine, but CUDA and HIP do not
      if (myMem == memDevice) {
        #if   defined(YAKL_ARCH_SYCL)
          stream.get_real_stream().fill<T>( arr.data() , val , arr.totElems() );
        #else
          c::parallel_for( "YAKL_internal_memset" , arr.totElems() , YAKL_LAMBDA (int i) {
            arr.data()[i] = val;
          } , DefaultLaunchConfig().set_stream(stream) );
        #endif
      } else if (myMem == memHost) {
        std::fill( arr.data(), arr.data()+arr.totElems(), val );
      }
    }
    #if defined(YAKL_AUTO_FENCE)
      fence();
    #endif
  }


  /** @private */
  template <class T, int rank, class B0, class B1, class B2, class B3, class I>
  YAKL_INLINE void memset( FSArray<T,rank,B0,B1,B2,B3> &arr , I val ) {
    for (index_t i = 0; i < arr.totElems(); i++) {
      arr.data()[i] = val;
    }
  }


  /** @private */
  template <class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3, class I>
  YAKL_INLINE void memset( SArray<T,rank,D0,D1,D2,D3> &arr , I val ) {
    for (index_t i = 0; i < arr.totElems(); i++) {
      arr.data()[i] = val;
    }
  }

}


