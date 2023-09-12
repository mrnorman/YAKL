
#pragma once
// Included by YAKL_intrinsics.h

__YAKL_NAMESPACE_WRAPPER_BEGIN__
namespace yakl {
  namespace intrinsics {

    template <class T1, class T2,
              typename std::enable_if<std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value,bool>::type=false>
    YAKL_INLINE decltype(T1()+T2()) merge(T1 const t, T2 const f, bool cond) { return cond ? t : f; }

    template <class T1, class T2, int rank, int myStyle>
    inline Array<decltype(T1()+T2()),rank,memHost,myStyle>
    merge( Array<T1  ,rank,memHost,myStyle> const & arr_true  ,
           Array<T2  ,rank,memHost,myStyle> const & arr_false ,
           Array<bool,rank,memHost,myStyle> const & mask      ) {
      #ifdef YAKL_DEBUG
        using yakl::componentwise::operator==;
        using yakl::componentwise::operator&&;
        using yakl::componentwise::operator!;
        if (!allocated(arr_true )) yakl_throw("ERROR: calling merge with arr_true  unallocated.");
        if (!allocated(arr_false)) yakl_throw("ERROR: calling merge with arr_false unallocated.");
        if (!allocated(mask     )) yakl_throw("ERROR: calling merge with mask      unallocated.");
        if (any( !( (shape(arr_true) == shape(arr_false)) && (shape(arr_false) == shape(mask)) ) ))
          yakl_throw("ERROR: calling merge with array shapes that do not match");
      #endif
      Array<decltype(T1()+T2()),rank,memHost,myStyle> ret = arr_true.createHostObject();
      for (index_t i=0; i < arr_true.totElems(); i++) {
        ret.data()[i] = mask.data()[i] ? arr_true.data()[i] : arr_false.data()[i];
      }
      return ret;
    }

    template <class T1, class T2, int rank, int myStyle>
    inline Array<decltype(T1()+T2()),rank,memDevice,myStyle>
    merge( Array<T1  ,rank,memDevice,myStyle> const & arr_true  ,
           Array<T2  ,rank,memDevice,myStyle> const & arr_false ,
           Array<bool,rank,memDevice,myStyle> const & mask      , Stream stream = Stream() ) {
      #ifdef YAKL_DEBUG
        using yakl::componentwise::operator==;
        using yakl::componentwise::operator&&;
        using yakl::componentwise::operator!;
        if (!allocated(arr_true )) yakl_throw("ERROR: calling merge with arr_true  unallocated.");
        if (!allocated(arr_false)) yakl_throw("ERROR: calling merge with arr_false unallocated.");
        if (!allocated(mask     )) yakl_throw("ERROR: calling merge with mask      unallocated.");
        if (any( !( (shape(arr_true) == shape(arr_false)) && (shape(arr_false) == shape(mask)) ) ))
          yakl_throw("ERROR: calling merge with array shapes that do not match");
      #endif
      Array<decltype(T1()+T2()),rank,memDevice,myStyle> ret = arr_true.createDeviceObject();
      c::parallel_for( "YAKL_internal_merge" , arr_true.totElems() , YAKL_LAMBDA (int i) {
        ret.data()[i] = mask.data()[i] ? arr_true.data()[i] : arr_false.data()[i];
      }, DefaultLaunchConfig().set_stream(stream) );
      ret.add_stream_dependency(stream);
      return ret;
    }

    template <class T1, class T2, int rank, index_t D0, index_t D1, index_t D2, index_t D3>
    YAKL_INLINE SArray<decltype(T1()+T2()),rank,D0,D1,D2,D3>
    merge( SArray<T1  ,rank,D0,D1,D2,D3> const & arr_true  ,
           SArray<T2  ,rank,D0,D1,D2,D3> const & arr_false ,
           SArray<bool,rank,D0,D1,D2,D3> const & mask      ) {
      SArray<decltype(T1()+T2()),rank,D0,D1,D2,D3> ret;
      for (index_t i=0; i < arr_true.totElems(); i++) {
        ret.data()[i] = mask.data()[i] ? arr_true.data()[i] : arr_false.data()[i];
      }
      return ret;
    }

    template <class T1, class T2, int rank, class B0, class B1, class B2, class B3>
    YAKL_INLINE FSArray<decltype(T1()+T2()),rank,B0,B1,B2,B3>
    merge( FSArray<T1  ,rank,B0,B1,B2,B3> const & arr_true  ,
           FSArray<T2  ,rank,B0,B1,B2,B3> const & arr_false ,
           FSArray<bool,rank,B0,B1,B2,B3> const & mask      ) {
      FSArray<decltype(T1()+T2()),rank,B0,B1,B2,B3> ret;
      for (index_t i=0; i < arr_true.totElems(); i++) {
        ret.data()[i] = mask.data()[i] ? arr_true.data()[i] : arr_false.data()[i];
      }
      return ret;
    }

  }
}
__YAKL_NAMESPACE_WRAPPER_END__

