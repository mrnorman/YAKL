
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T1, class T2,
              typename std::enable_if<std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value,bool>::type=false>
    KOKKOS_INLINE_FUNCTION decltype(T1()+T2()) merge(T1 const t, T2 const f, bool cond) { return cond ? t : f; }

    template <class T1, class T2, int rank, int myStyle>
    inline Array<decltype(T1()+T2()),rank,memHost,myStyle>
    merge( Array<T1  ,rank,memHost,myStyle> const & arr_true  ,
           Array<T2  ,rank,memHost,myStyle> const & arr_false ,
           Array<bool,rank,memHost,myStyle> const & mask      ) {
      #ifdef KOKKOS_ENABLE_DEBUG
        using yakl::componentwise::operator==;
        using yakl::componentwise::operator&&;
        using yakl::componentwise::operator!;
        if (!allocated(arr_true )) Kokkos::abort("ERROR: calling merge with arr_true  unallocated.");
        if (!allocated(arr_false)) Kokkos::abort("ERROR: calling merge with arr_false unallocated.");
        if (!allocated(mask     )) Kokkos::abort("ERROR: calling merge with mask      unallocated.");
        if (any( !( (shape(arr_true) == shape(arr_false)) && (shape(arr_false) == shape(mask)) ) ))
          Kokkos::abort("ERROR: calling merge with array shapes that do not match");
      #endif
      Array<decltype(T1()+T2()),rank,memHost,myStyle> ret = arr_true.createHostObject();
      for (size_t i=0; i < arr_true.totElems(); i++) {
        ret.data()[i] = mask.data()[i] ? arr_true.data()[i] : arr_false.data()[i];
      }
      return ret;
    }

    template <class T1, class T2, int rank, int myStyle>
    inline Array<decltype(T1()+T2()),rank,memDevice,myStyle>
    merge( Array<T1  ,rank,memDevice,myStyle> const & arr_true  ,
           Array<T2  ,rank,memDevice,myStyle> const & arr_false ,
           Array<bool,rank,memDevice,myStyle> const & mask      ) {
      #ifdef KOKKOS_ENABLE_DEBUG
        using yakl::componentwise::operator==;
        using yakl::componentwise::operator&&;
        using yakl::componentwise::operator!;
        if (!allocated(arr_true )) Kokkos::abort("ERROR: calling merge with arr_true  unallocated.");
        if (!allocated(arr_false)) Kokkos::abort("ERROR: calling merge with arr_false unallocated.");
        if (!allocated(mask     )) Kokkos::abort("ERROR: calling merge with mask      unallocated.");
        if (any( !( (shape(arr_true) == shape(arr_false)) && (shape(arr_false) == shape(mask)) ) ))
          Kokkos::abort("ERROR: calling merge with array shapes that do not match");
      #endif
      Array<decltype(T1()+T2()),rank,memDevice,myStyle> ret = arr_true.createDeviceObject();
      c::parallel_for( YAKL_AUTO_LABEL() , arr_true.totElems() , KOKKOS_LAMBDA (int i) {
        ret.data()[i] = mask.data()[i] ? arr_true.data()[i] : arr_false.data()[i];
      });
      return ret;
    }

    template <class T1, class T2, int rank, size_t D0, size_t D1, size_t D2, size_t D3>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()+T2()),rank,D0,D1,D2,D3>
    merge( SArray<T1  ,rank,D0,D1,D2,D3> const & arr_true  ,
           SArray<T2  ,rank,D0,D1,D2,D3> const & arr_false ,
           SArray<bool,rank,D0,D1,D2,D3> const & mask      ) {
      SArray<decltype(T1()+T2()),rank,D0,D1,D2,D3> ret;
      for (size_t i=0; i < arr_true.totElems(); i++) {
        ret.data()[i] = mask.data()[i] ? arr_true.data()[i] : arr_false.data()[i];
      }
      return ret;
    }

    template <class T1, class T2, int rank, class B0, class B1, class B2, class B3>
    KOKKOS_INLINE_FUNCTION FSArray<decltype(T1()+T2()),rank,B0,B1,B2,B3>
    merge( FSArray<T1  ,rank,B0,B1,B2,B3> const & arr_true  ,
           FSArray<T2  ,rank,B0,B1,B2,B3> const & arr_false ,
           FSArray<bool,rank,B0,B1,B2,B3> const & mask      ) {
      FSArray<decltype(T1()+T2()),rank,B0,B1,B2,B3> ret;
      for (size_t i=0; i < arr_true.totElems(); i++) {
        ret.data()[i] = mask.data()[i] ? arr_true.data()[i] : arr_false.data()[i];
      }
      return ret;
    }

  }
}

