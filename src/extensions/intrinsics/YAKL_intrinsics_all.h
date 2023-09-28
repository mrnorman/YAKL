
#pragma once
// Included by YAKL_intrinsics.h

__YAKL_NAMESPACE_WRAPPER_BEGIN__
namespace yakl {
  namespace intrinsics {

    template <class T, int rank, int myStyle>
    inline bool all( Array<T,rank,memHost,myStyle> arr ) {
      #ifdef YAKL_DEBUG
        if (!arr.initialized()) { yakl_throw("ERROR: calling all on an array that has not been initialized"); }
      #endif
      bool all_true = true;
      for (int i=0; i < arr.totElems(); i++) { if (!arr.data()[i]) all_true = false; }
      return all_true;
    }

    template <class T, int rank, int myStyle>
    inline bool all( Array<T,rank,memDevice,myStyle> arr , Stream stream = Stream() ) {
      #ifdef YAKL_DEBUG
        if (!arr.initialized()) { yakl_throw("ERROR: calling all on an array that has not been initialized"); }
      #endif
      ScalarLiveOut<bool> all_true(true,stream);
      c::parallel_for( "YAKL_internal_all" , arr.totElems() , YAKL_LAMBDA (int i) { if (!arr.data()[i]) all_true = false; },
                       DefaultLaunchConfig().set_stream(stream) );
      return all_true.hostRead(stream);
    }

    template <class T, int rank, index_t D0, index_t D1, index_t D2, index_t D3>
    inline bool all( SArray<T,rank,D0,D1,D2,D3> const &arr ) {
      bool all_true = true;
      for (int i=0; i < arr.totElems(); i++) { if (!arr.data()[i]) all_true = false; }
      return all_true;
    }

    template <class T, int rank, class B0, class B1, class B2, class B3>
    inline bool all( FSArray<T,rank,B0,B1,B2,B3> const &arr ) {
      bool all_true = true;
      for (int i=0; i < arr.totElems(); i++) { if (!arr.data()[i]) all_true = false; }
      return all_true;
    }

  }
}
__YAKL_NAMESPACE_WRAPPER_END__

