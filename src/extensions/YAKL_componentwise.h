
#pragma once
// Included by YAKL.h

namespace yakl {
  // These are some convenient intrinsics functions (think Fortran intrinsics library)

  // Componentwise operations on arrays

  // You're going to see some strange things in here when using parallel_for. It's all because of CUDA. Thanks CUDA.

  /** @brief This namespace contains routines that perform element-wise / component-wise operations on `Array`, `SArray`, and `FSArray` objects. */
  namespace componentwise {

    ///////////////////////////////////////////////////////////////////////
    // Binary operators with Array LHS and scalar RHS
    ///////////////////////////////////////////////////////////////////////

    // Addition
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()+T2()),N,memHost,STYLE>
    operator+( Array<T1,N,memHost,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createHostObject<decltype(T1()+T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] + right; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()+T2()),N,memDevice,STYLE>
    add_array_scalar( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()+T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator+" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] + right; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()+T2()),N,memDevice,STYLE>
    operator+( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      return add_array_scalar( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()+T2()),N,D0,D1,D2,D3>
    operator+( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()+T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] + right; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()+T2()),N,B0,B1,B2,B3>
    operator+( FSArray<T1,N,B0,B1,B2,B3> const &left , T2 const &right ) {
      FSArray<decltype(T1()+T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] + right; }
      return ret;
    }

    // Subtraction
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()-T2()),N,memHost,STYLE>
    operator-( Array<T1,N,memHost,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createHostObject<decltype(T1()-T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] - right; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()-T2()),N,memDevice,STYLE>
    sub_array_scalar( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()-T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator-" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] - right; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()-T2()),N,memDevice,STYLE>
    operator-( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      return sub_array_scalar( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()-T2()),N,D0,D1,D2,D3>
    operator-( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()-T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] - right; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()-T2()),N,B0,B1,B2,B3>
    operator-( FSArray<T1,N,B0,B1,B2,B3> const &left , T2 const &right ) {
      FSArray<decltype(T1()-T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] - right; }
      return ret;
    }

    // Multiplication
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()*T2()),N,memHost,STYLE>
    operator*( Array<T1,N,memHost,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createHostObject<decltype(T1()*T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] * right; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()*T2()),N,memDevice,STYLE>
    mult_array_scalar( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()*T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator*" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] * right; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()*T2()),N,memDevice,STYLE>
    operator*( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      return mult_array_scalar( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()*T2()),N,D0,D1,D2,D3>
    operator*( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()*T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] * right; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()*T2()),N,B0,B1,B2,B3>
    operator*( FSArray<T1,N,B0,B1,B2,B3> const &left , T2 const &right ) {
      FSArray<decltype(T1()*T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] * right; }
      return ret;
    }

    // Division
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()/T2()),N,memHost,STYLE>
    operator/( Array<T1,N,memHost,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createHostObject<decltype(T1()/T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] / right; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()/T2()),N,memDevice,STYLE>
    div_array_scalar( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()/T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator/" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] / right; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()/T2()),N,memDevice,STYLE>
    operator/( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      return div_array_scalar( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()/T2()),N,D0,D1,D2,D3>
    operator/( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()/T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] / right; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()/T2()),N,B0,B1,B2,B3>
    operator/( FSArray<T1,N,B0,B1,B2,B3> const &left , T2 const &right ) {
      FSArray<decltype(T1()/T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] / right; }
      return ret;
    }

    // Greater than >
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()>T2()),N,memHost,STYLE>
    operator>( Array<T1,N,memHost,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createHostObject<decltype(T1()>T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] > right; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()>T2()),N,memDevice,STYLE>
    gt_array_scalar( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()>T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator>" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] > right; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()>T2()),N,memDevice,STYLE>
    operator>( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      return gt_array_scalar( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()>T2()),N,D0,D1,D2,D3>
    operator>( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()>T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] > right; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()>T2()),N,B0,B1,B2,B3>
    operator>( FSArray<T1,N,B0,B1,B2,B3> const &left , T2 const &right ) {
      FSArray<decltype(T1()>T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] > right; }
      return ret;
    }

    // Less than <
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()<T2()),N,memHost,STYLE>
    operator<( Array<T1,N,memHost,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createHostObject<decltype(T1()<T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] < right; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()<T2()),N,memDevice,STYLE>
    lt_array_scalar( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()<T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator<" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] < right; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()<T2()),N,memDevice,STYLE>
    operator<( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      return lt_array_scalar( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()<T2()),N,D0,D1,D2,D3>
    operator<( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()<T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] < right; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()<T2()),N,B0,B1,B2,B3>
    operator<( FSArray<T1,N,B0,B1,B2,B3> const &left , T2 const &right ) {
      FSArray<decltype(T1()<T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] < right; }
      return ret;
    }

    // Greater than or equal to >=
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()>=T2()),N,memHost,STYLE>
    operator>=( Array<T1,N,memHost,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createHostObject<decltype(T1()>=T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] >= right; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()>=T2()),N,memDevice,STYLE>
    ge_array_scalar( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()>=T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator>=" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] >= right; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()>=T2()),N,memDevice,STYLE>
    operator>=( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      return ge_array_scalar( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()>=T2()),N,D0,D1,D2,D3>
    operator>=( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()>=T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] >= right; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()>=T2()),N,B0,B1,B2,B3>
    operator>=( FSArray<T1,N,B0,B1,B2,B3> const &left , T2 const &right ) {
      FSArray<decltype(T1()>=T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] >= right; }
      return ret;
    }

    // Less than or equal to <=
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()<=T2()),N,memHost,STYLE>
    operator<=( Array<T1,N,memHost,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createHostObject<decltype(T1()<=T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] <= right; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()<=T2()),N,memDevice,STYLE>
    le_array_scalar( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()<=T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator<=" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] <= right; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()<=T2()),N,memDevice,STYLE>
    operator<=( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      return le_array_scalar( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()<=T2()),N,D0,D1,D2,D3>
    operator<=( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()<=T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] <= right; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()<=T2()),N,B0,B1,B2,B3>
    operator<=( FSArray<T1,N,B0,B1,B2,B3> const &left , T2 const &right ) {
      FSArray<decltype(T1()<=T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] <= right; }
      return ret;
    }

    // Equal to ==
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()==T2()),N,memHost,STYLE>
    operator==( Array<T1,N,memHost,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createHostObject<decltype(T1()==T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] == right; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()==T2()),N,memDevice,STYLE>
    eq_array_scalar( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()==T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator==" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] == right; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()==T2()),N,memDevice,STYLE>
    operator==( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      return eq_array_scalar( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()==T2()),N,D0,D1,D2,D3>
    operator==( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()==T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] == right; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()==T2()),N,B0,B1,B2,B3>
    operator==( FSArray<T1,N,B0,B1,B2,B3> const &left , T2 const &right ) {
      FSArray<decltype(T1()==T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] == right; }
      return ret;
    }

    // Not equal to !=
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()!=T2()),N,memHost,STYLE>
    operator!=( Array<T1,N,memHost,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createHostObject<decltype(T1()!=T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] != right; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()!=T2()),N,memDevice,STYLE>
    ne_array_scalar( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()!=T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator!=" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] != right; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()!=T2()),N,memDevice,STYLE>
    operator!=( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      return ne_array_scalar( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()!=T2()),N,D0,D1,D2,D3>
    operator!=( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()!=T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] != right; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()!=T2()),N,B0,B1,B2,B3>
    operator!=( FSArray<T1,N,B0,B1,B2,B3> const &left , T2 const &right ) {
      FSArray<decltype(T1()!=T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] != right; }
      return ret;
    }

    // logical and &&
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()&&T2()),N,memHost,STYLE>
    operator&&( Array<T1,N,memHost,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createHostObject<decltype(T1()&&T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] && right; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()&&T2()),N,memDevice,STYLE>
    and_array_scalar( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()&&T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator&&" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] && right; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()&&T2()),N,memDevice,STYLE>
    operator&&( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      return and_array_scalar( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()&&T2()),N,D0,D1,D2,D3>
    operator&&( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()&&T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] && right; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()&&T2()),N,B0,B1,B2,B3>
    operator&&( FSArray<T1,N,B0,B1,B2,B3> const &left , T2 const &right ) {
      FSArray<decltype(T1()&&T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] && right; }
      return ret;
    }

    // logical or ||
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()||T2()),N,memHost,STYLE>
    operator||( Array<T1,N,memHost,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createHostObject<decltype(T1()||T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] || right; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()||T2()),N,memDevice,STYLE>
    or_array_scalar( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()||T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator||" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] || right; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    inline Array<decltype(T1()||T2()),N,memDevice,STYLE>
    operator||( Array<T1,N,memDevice,STYLE> const &left , T2 const &right ) {
      return or_array_scalar( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()||T2()),N,D0,D1,D2,D3>
    operator||( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()||T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] || right; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()||T2()),N,B0,B1,B2,B3>
    operator||( FSArray<T1,N,B0,B1,B2,B3> const &left , T2 const &right ) {
      FSArray<decltype(T1()||T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] || right; }
      return ret;
    }


    ///////////////////////////////////////////////////////////////////////
    // Binary operators with scalar LHS and Array RHS
    ///////////////////////////////////////////////////////////////////////

    // Addition
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()+T2()),N,memHost,STYLE>
    operator+( T1 const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = right.template createHostObject<decltype(T1()+T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left + right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()+T2()),N,memDevice,STYLE>
    add_scalar_array( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = right.template createDeviceObject<decltype(T1()+T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator+" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left + right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()+T2()),N,memDevice,STYLE>
    operator+( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      return add_scalar_array( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()+T2()),N,D0,D1,D2,D3>
    operator+( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()+T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left + right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()+T2()),N,B0,B1,B2,B3>
    operator+( T1 const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()+T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left + right.data()[i]; }
      return ret;
    }

    // Subtraction
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()-T2()),N,memHost,STYLE>
    operator-( T1 const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = right.template createHostObject<decltype(T1()-T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left - right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()-T2()),N,memDevice,STYLE>
    sub_scalar_array( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = right.template createDeviceObject<decltype(T1()-T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator-" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left - right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()-T2()),N,memDevice,STYLE>
    operator-( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      return sub_scalar_array( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()-T2()),N,D0,D1,D2,D3>
    operator-( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()-T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left - right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()-T2()),N,B0,B1,B2,B3>
    operator-( T1 const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()-T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left - right.data()[i]; }
      return ret;
    }

    // Multiplication
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()*T2()),N,memHost,STYLE>
    operator*( T1 const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = right.template createHostObject<decltype(T1()*T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left * right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()*T2()),N,memDevice,STYLE>
    mult_scalar_array( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = right.template createDeviceObject<decltype(T1()*T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator*" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left * right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()*T2()),N,memDevice,STYLE>
    operator*( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      return mult_scalar_array( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()*T2()),N,D0,D1,D2,D3>
    operator*( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()*T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left * right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()*T2()),N,B0,B1,B2,B3>
    operator*( T1 const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()*T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left * right.data()[i]; }
      return ret;
    }

    // Division
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()/T2()),N,memHost,STYLE>
    operator/( T1 const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = right.template createHostObject<decltype(T1()/T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left / right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()/T2()),N,memDevice,STYLE>
    div_scalar_array( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = right.template createDeviceObject<decltype(T1()/T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator/" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left / right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()/T2()),N,memDevice,STYLE>
    operator/( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      return div_scalar_array( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()/T2()),N,D0,D1,D2,D3>
    operator/( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()/T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left / right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()/T2()),N,B0,B1,B2,B3>
    operator/( T1 const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()/T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left / right.data()[i]; }
      return ret;
    }

    // Greater than >
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()>T2()),N,memHost,STYLE>
    operator>( T1 const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = right.template createHostObject<decltype(T1()>T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left > right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()>T2()),N,memDevice,STYLE>
    gt_scalar_array( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = right.template createDeviceObject<decltype(T1()>T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator>" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left > right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()>T2()),N,memDevice,STYLE>
    operator>( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      return gt_scalar_array( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()>T2()),N,D0,D1,D2,D3>
    operator>( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()>T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left > right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()>T2()),N,B0,B1,B2,B3>
    operator>( T1 const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()>T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left > right.data()[i]; }
      return ret;
    }

    // Less than <
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()<T2()),N,memHost,STYLE>
    operator<( T1 const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = right.template createHostObject<decltype(T1()<T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left < right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()<T2()),N,memDevice,STYLE>
    lt_scalar_array( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = right.template createDeviceObject<decltype(T1()<T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator<" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left < right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()<T2()),N,memDevice,STYLE>
    operator<( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      return lt_scalar_array( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()<T2()),N,D0,D1,D2,D3>
    operator<( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()<T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left < right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()<T2()),N,B0,B1,B2,B3>
    operator<( T1 const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()<T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left < right.data()[i]; }
      return ret;
    }

    // Greater than or equal to >=
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()>=T2()),N,memHost,STYLE>
    operator>=( T1 const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = right.template createHostObject<decltype(T1()>=T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left >= right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()>=T2()),N,memDevice,STYLE>
    ge_scalar_array( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = right.template createDeviceObject<decltype(T1()>=T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator>=" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left >= right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()>=T2()),N,memDevice,STYLE>
    operator>=( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      return ge_scalar_array( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()>=T2()),N,D0,D1,D2,D3>
    operator>=( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()>=T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left >= right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()>=T2()),N,B0,B1,B2,B3>
    operator>=( T1 const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()>=T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left >= right.data()[i]; }
      return ret;
    }

    // Less than or equal to <=
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()<=T2()),N,memHost,STYLE>
    operator<=( T1 const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = right.template createHostObject<decltype(T1()<=T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left <= right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()<=T2()),N,memDevice,STYLE>
    le_scalar_array( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = right.template createDeviceObject<decltype(T1()<=T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator<=" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left <= right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()<=T2()),N,memDevice,STYLE>
    operator<=( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      return le_scalar_array( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()<=T2()),N,D0,D1,D2,D3>
    operator<=( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()<=T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left <= right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()<=T2()),N,B0,B1,B2,B3>
    operator<=( T1 const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()<=T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left <= right.data()[i]; }
      return ret;
    }

    // Equal to ==
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()==T2()),N,memHost,STYLE>
    operator==( T1 const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = right.template createHostObject<decltype(T1()==T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left == right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()==T2()),N,memDevice,STYLE>
    eq_scalar_array( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = right.template createDeviceObject<decltype(T1()==T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator==" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left == right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()==T2()),N,memDevice,STYLE>
    operator==( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      return eq_scalar_array( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()==T2()),N,D0,D1,D2,D3>
    operator==( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()==T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left == right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()==T2()),N,B0,B1,B2,B3>
    operator==( T1 const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()==T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left == right.data()[i]; }
      return ret;
    }

    // Not equal to !=
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()!=T2()),N,memHost,STYLE>
    operator!=( T1 const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = right.template createHostObject<decltype(T1()!=T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left != right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()!=T2()),N,memDevice,STYLE>
    ne_scalar_array( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = right.template createDeviceObject<decltype(T1()!=T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator!=" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left != right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()!=T2()),N,memDevice,STYLE>
    operator!=( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      return ne_scalar_array( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()!=T2()),N,D0,D1,D2,D3>
    operator!=( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()!=T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left != right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()!=T2()),N,B0,B1,B2,B3>
    operator!=( T1 const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()!=T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left != right.data()[i]; }
      return ret;
    }

    // logical and &&
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()&&T2()),N,memHost,STYLE>
    operator&&( T1 const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = right.template createHostObject<decltype(T1()&&T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left && right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()&&T2()),N,memDevice,STYLE>
    and_scalar_array( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = right.template createDeviceObject<decltype(T1()&&T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator&&" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left && right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()&&T2()),N,memDevice,STYLE>
    operator&&( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      return and_scalar_array( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()&&T2()),N,D0,D1,D2,D3>
    operator&&( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()&&T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left && right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()&&T2()),N,B0,B1,B2,B3>
    operator&&( T1 const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()&&T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left && right.data()[i]; }
      return ret;
    }

    // logical or ||
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()||T2()),N,memHost,STYLE>
    operator||( T1 const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = right.template createHostObject<decltype(T1()||T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left || right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()||T2()),N,memDevice,STYLE>
    or_scalar_array( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = right.template createDeviceObject<decltype(T1()||T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator||" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left || right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    inline Array<decltype(T1()||T2()),N,memDevice,STYLE>
    operator||( T1 const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      return or_scalar_array( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE SArray<decltype(T1()||T2()),N,D0,D1,D2,D3>
    operator||( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()||T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left || right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    YAKL_INLINE FSArray<decltype(T1()||T2()),N,B0,B1,B2,B3>
    operator||( T1 const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()||T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left || right.data()[i]; }
      return ret;
    }


    ///////////////////////////////////////////////////////////////////////
    // Binary operators with Array LHS and Array RHS
    ///////////////////////////////////////////////////////////////////////

    // Addition
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()+T2()),N,memHost,STYLE>
    operator+( Array<T1,N,memHost,STYLE> const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = left.template createHostObject<decltype(T1()+T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] + right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()+T2()),N,memDevice,STYLE>
    operator+( Array<T1,N,memDevice,STYLE> const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()+T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator+" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] + right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
    YAKL_INLINE SArray<decltype(T1()+T2()),N,D0,D1,D2,D3>
    operator+( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()+T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] + right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3>
    YAKL_INLINE FSArray<decltype(T1()+T2()),N,B0,B1,B2,B3>
    operator+( FSArray<T1,N,B0,B1,B2,B3> const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()+T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] + right.data()[i]; }
      return ret;
    }

    // Subtraction
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()-T2()),N,memHost,STYLE>
    operator-( Array<T1,N,memHost,STYLE> const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = left.template createHostObject<decltype(T1()-T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] - right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()-T2()),N,memDevice,STYLE>
    operator-( Array<T1,N,memDevice,STYLE> const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()-T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator-" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] - right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
    YAKL_INLINE SArray<decltype(T1()-T2()),N,D0,D1,D2,D3>
    operator-( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()-T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] - right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3>
    YAKL_INLINE FSArray<decltype(T1()-T2()),N,B0,B1,B2,B3>
    operator-( FSArray<T1,N,B0,B1,B2,B3> const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()-T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] - right.data()[i]; }
      return ret;
    }

    // Multiplication
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()*T2()),N,memHost,STYLE>
    operator*( Array<T1,N,memHost,STYLE> const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = left.template createHostObject<decltype(T1()*T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] * right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()*T2()),N,memDevice,STYLE>
    operator*( Array<T1,N,memDevice,STYLE> const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()*T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator*" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] * right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
    YAKL_INLINE SArray<decltype(T1()*T2()),N,D0,D1,D2,D3>
    operator*( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()*T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] * right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3>
    YAKL_INLINE FSArray<decltype(T1()*T2()),N,B0,B1,B2,B3>
    operator*( FSArray<T1,N,B0,B1,B2,B3> const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()*T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] * right.data()[i]; }
      return ret;
    }

    // Division
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()/T2()),N,memHost,STYLE>
    operator/( Array<T1,N,memHost,STYLE> const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = left.template createHostObject<decltype(T1()/T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] / right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()/T2()),N,memDevice,STYLE>
    operator/( Array<T1,N,memDevice,STYLE> const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()/T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator/" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] / right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
    YAKL_INLINE SArray<decltype(T1()/T2()),N,D0,D1,D2,D3>
    operator/( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()/T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] / right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3>
    YAKL_INLINE FSArray<decltype(T1()/T2()),N,B0,B1,B2,B3>
    operator/( FSArray<T1,N,B0,B1,B2,B3> const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()/T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] / right.data()[i]; }
      return ret;
    }

    // Greater than >
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()>T2()),N,memHost,STYLE>
    operator>( Array<T1,N,memHost,STYLE> const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = left.template createHostObject<decltype(T1()>T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] > right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()>T2()),N,memDevice,STYLE>
    operator>( Array<T1,N,memDevice,STYLE> const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()>T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator>" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] > right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
    YAKL_INLINE SArray<decltype(T1()>T2()),N,D0,D1,D2,D3>
    operator>( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()>T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] > right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3>
    YAKL_INLINE FSArray<decltype(T1()>T2()),N,B0,B1,B2,B3>
    operator>( FSArray<T1,N,B0,B1,B2,B3> const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()>T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] > right.data()[i]; }
      return ret;
    }

    // Less than <
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()<T2()),N,memHost,STYLE>
    operator<( Array<T1,N,memHost,STYLE> const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = left.template createHostObject<decltype(T1()<T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] < right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<bool,N,memDevice,STYLE>
    lt_array_array( Array<T1,N,memDevice,STYLE> const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = left.template createDeviceObject<bool>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator<" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] < right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<bool,N,memDevice,STYLE>
    operator<( Array<T1,N,memDevice,STYLE> const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      return lt_array_array( left , right );
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
    YAKL_INLINE SArray<decltype(T1()<T2()),N,D0,D1,D2,D3>
    operator<( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()<T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] < right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3>
    YAKL_INLINE FSArray<decltype(T1()<T2()),N,B0,B1,B2,B3>
    operator<( FSArray<T1,N,B0,B1,B2,B3> const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()<T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] < right.data()[i]; }
      return ret;
    }

    // Greater than or equal to >=
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()>=T2()),N,memHost,STYLE>
    operator>=( Array<T1,N,memHost,STYLE> const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = left.template createHostObject<decltype(T1()>=T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] >= right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()>=T2()),N,memDevice,STYLE>
    operator>=( Array<T1,N,memDevice,STYLE> const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()>=T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator>=" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] >= right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
    YAKL_INLINE SArray<decltype(T1()>=T2()),N,D0,D1,D2,D3>
    operator>=( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()>=T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] >= right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3>
    YAKL_INLINE FSArray<decltype(T1()>=T2()),N,B0,B1,B2,B3>
    operator>=( FSArray<T1,N,B0,B1,B2,B3> const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()>=T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] >= right.data()[i]; }
      return ret;
    }

    // Less than or equal to <=
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()<=T2()),N,memHost,STYLE>
    operator<=( Array<T1,N,memHost,STYLE> const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = left.template createHostObject<decltype(T1()<=T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] <= right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()<=T2()),N,memDevice,STYLE>
    operator<=( Array<T1,N,memDevice,STYLE> const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()<=T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator<=" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] <= right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
    YAKL_INLINE SArray<decltype(T1()<=T2()),N,D0,D1,D2,D3>
    operator<=( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()<=T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] <= right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3>
    YAKL_INLINE FSArray<decltype(T1()<=T2()),N,B0,B1,B2,B3>
    operator<=( FSArray<T1,N,B0,B1,B2,B3> const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()<=T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] <= right.data()[i]; }
      return ret;
    }

    // Equal to ==
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()==T2()),N,memHost,STYLE>
    operator==( Array<T1,N,memHost,STYLE> const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = left.template createHostObject<decltype(T1()==T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] == right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()==T2()),N,memDevice,STYLE>
    operator==( Array<T1,N,memDevice,STYLE> const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()==T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator==" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] == right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
    YAKL_INLINE SArray<decltype(T1()==T2()),N,D0,D1,D2,D3>
    operator==( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()==T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] == right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3>
    YAKL_INLINE FSArray<decltype(T1()==T2()),N,B0,B1,B2,B3>
    operator==( FSArray<T1,N,B0,B1,B2,B3> const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()==T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] == right.data()[i]; }
      return ret;
    }

    // Not equal to !=
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()!=T2()),N,memHost,STYLE>
    operator!=( Array<T1,N,memHost,STYLE> const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = left.template createHostObject<decltype(T1()!=T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] != right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()!=T2()),N,memDevice,STYLE>
    operator!=( Array<T1,N,memDevice,STYLE> const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()!=T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator!=" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] != right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
    YAKL_INLINE SArray<decltype(T1()!=T2()),N,D0,D1,D2,D3>
    operator!=( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()!=T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] != right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3>
    YAKL_INLINE FSArray<decltype(T1()!=T2()),N,B0,B1,B2,B3>
    operator!=( FSArray<T1,N,B0,B1,B2,B3> const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()!=T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] != right.data()[i]; }
      return ret;
    }

    // logical and &&
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()&&T2()),N,memHost,STYLE>
    operator&&( Array<T1,N,memHost,STYLE> const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = left.template createHostObject<decltype(T1()&&T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] && right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()&&T2()),N,memDevice,STYLE>
    operator&&( Array<T1,N,memDevice,STYLE> const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()&&T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator&&" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] && right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
    YAKL_INLINE SArray<decltype(T1()&&T2()),N,D0,D1,D2,D3>
    operator&&( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()&&T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] && right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3>
    YAKL_INLINE FSArray<decltype(T1()&&T2()),N,B0,B1,B2,B3>
    operator&&( FSArray<T1,N,B0,B1,B2,B3> const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()&&T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] && right.data()[i]; }
      return ret;
    }

    // logical or ||
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()||T2()),N,memHost,STYLE>
    operator||( Array<T1,N,memHost,STYLE> const &left , Array<T2,N,memHost,STYLE> const &right ) {
      auto ret = left.template createHostObject<decltype(T1()||T2())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] || right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, int STYLE>
    inline Array<decltype(T1()||T2()),N,memDevice,STYLE>
    operator||( Array<T1,N,memDevice,STYLE> const &left , Array<T2,N,memDevice,STYLE> const &right ) {
      auto ret = left.template createDeviceObject<decltype(T1()||T2())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator||" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i] || right.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, class T2, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
    YAKL_INLINE SArray<decltype(T1()||T2()),N,D0,D1,D2,D3>
    operator||( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()||T2()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] || right.data()[i]; }
      return ret;
    }
    template <class T1, class T2, int N, class B0, class B1, class B2, class B3>
    YAKL_INLINE FSArray<decltype(T1()||T2()),N,B0,B1,B2,B3>
    operator||( FSArray<T1,N,B0,B1,B2,B3> const &left , FSArray<T2,N,B0,B1,B2,B3> const &right ) {
      FSArray<decltype(T1()||T2()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] || right.data()[i]; }
      return ret;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // Unary operators
    ///////////////////////////////////////////////////////////////////////////////////////////

    // logical not !
    template <class T1, int N, int STYLE>
    inline Array<decltype(!T1()),N,memHost,STYLE>
    operator!( Array<T1,N,memHost,STYLE> const &left ) {
      auto ret = left.template createHostObject<decltype(!T1())>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = ! left.data()[i]; }
      return ret;
    }
    template <class T1, int N, int STYLE>
    inline Array<decltype(!T1()),N,memDevice,STYLE>
    operator!( Array<T1,N,memDevice,STYLE> const &left ) {
      auto ret = left.template createDeviceObject<decltype(!T1())>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator!" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = ! left.data()[i]; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
    YAKL_INLINE SArray<decltype(!T1()),N,D0,D1,D2,D3>
    operator!( SArray<T1,N,D0,D1,D2,D3> const &left ) {
      SArray<decltype(!T1()),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = ! left.data()[i]; }
      return ret;
    }
    template <class T1, int N, class B0, class B1, class B2, class B3>
    YAKL_INLINE FSArray<decltype(!T1()),N,B0,B1,B2,B3>
    operator!( FSArray<T1,N,B0,B1,B2,B3> const &left ) {
      FSArray<decltype(!T1()),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = ! left.data()[i]; }
      return ret;
    }

    // increment ++
    template <class T1, int N, int STYLE>
    inline Array<decltype(T1()+1),N,memHost,STYLE>
    operator++( Array<T1,N,memHost,STYLE> const &left ) {
      auto ret = left.template createHostObject<decltype(T1()+1)>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i]+1; }
      return ret;
    }
    template <class T1, int N, int STYLE>
    inline Array<decltype(T1()+1),N,memDevice,STYLE>
    operator++( Array<T1,N,memDevice,STYLE> const &left ) {
      auto ret = left.template createDeviceObject<decltype(T1()+1)>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator++" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i]+1; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
    YAKL_INLINE SArray<decltype(T1()+1),N,D0,D1,D2,D3>
    operator++( SArray<T1,N,D0,D1,D2,D3> const &left ) {
      SArray<decltype(T1()+1),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i]+1; }
      return ret;
    }
    template <class T1, int N, class B0, class B1, class B2, class B3>
    YAKL_INLINE FSArray<decltype(T1()+1),N,B0,B1,B2,B3>
    operator++( FSArray<T1,N,B0,B1,B2,B3> const &left ) {
      FSArray<decltype(T1()+1),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i]+1; }
      return ret;
    }

    // increment ++
    template <class T1, int N, int STYLE>
    inline Array<decltype(T1()+1),N,memHost,STYLE>
    operator++( Array<T1,N,memHost,STYLE> const &left , int dummy) {
      auto ret = left.template createHostObject<decltype(T1()+1)>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i]+1; }
      return ret;
    }
    template <class T1, int N, int STYLE>
    inline Array<decltype(T1()+1),N,memDevice,STYLE>
    operator++( Array<T1,N,memDevice,STYLE> const &left , int dummy) {
      auto ret = left.template createDeviceObject<decltype(T1()+1)>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator++" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i]+1; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
    YAKL_INLINE SArray<decltype(T1()+1),N,D0,D1,D2,D3>
    operator++( SArray<T1,N,D0,D1,D2,D3> const &left , int dummy) {
      SArray<decltype(T1()+1),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i]+1; }
      return ret;
    }
    template <class T1, int N, class B0, class B1, class B2, class B3>
    YAKL_INLINE FSArray<decltype(T1()+1),N,B0,B1,B2,B3>
    operator++( FSArray<T1,N,B0,B1,B2,B3> const &left , int dummy) {
      FSArray<decltype(T1()+1),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i]+1; }
      return ret;
    }

    // decrement --
    template <class T1, int N, int STYLE>
    inline Array<decltype(T1()-1),N,memHost,STYLE>
    operator--( Array<T1,N,memHost,STYLE> const &left ) {
      auto ret = left.template createHostObject<decltype(T1()-1)>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i]-1; }
      return ret;
    }
    template <class T1, int N, int STYLE>
    inline Array<decltype(T1()-1),N,memDevice,STYLE>
    operator--( Array<T1,N,memDevice,STYLE> const &left ) {
      auto ret = left.template createDeviceObject<decltype(T1()-1)>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator--" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i]-1; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
    YAKL_INLINE SArray<decltype(T1()-1),N,D0,D1,D2,D3>
    operator--( SArray<T1,N,D0,D1,D2,D3> const &left ) {
      SArray<decltype(T1()-1),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i]-1; }
      return ret;
    }
    template <class T1, int N, class B0, class B1, class B2, class B3>
    YAKL_INLINE FSArray<decltype(T1()-1),N,B0,B1,B2,B3>
    operator--( FSArray<T1,N,B0,B1,B2,B3> const &left ) {
      FSArray<decltype(T1()-1),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i]-1; }
      return ret;
    }

    // decrement --
    template <class T1, int N, int STYLE>
    inline Array<decltype(T1()-1),N,memHost,STYLE>
    operator--( Array<T1,N,memHost,STYLE> const &left , int dummy ) {
      auto ret = left.template createHostObject<decltype(T1()-1)>();
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i]-1; }
      return ret;
    }
    template <class T1, int N, int STYLE>
    inline Array<decltype(T1()-1),N,memDevice,STYLE>
    operator--( Array<T1,N,memDevice,STYLE> const &left , int dummy ) {
      auto ret = left.template createDeviceObject<decltype(T1()-1)>();
      if constexpr (streams_enabled) fence();
      c::parallel_for( "YAKL_internal_array_operator--" , ret.totElems() , YAKL_LAMBDA (int i) { ret.data()[i] = left.data()[i]-1; });
      if constexpr (streams_enabled) fence();
      return ret;
    }
    template <class T1, int N, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
    YAKL_INLINE SArray<decltype(T1()-1),N,D0,D1,D2,D3>
    operator--( SArray<T1,N,D0,D1,D2,D3> const &left , int dummy ) {
      SArray<decltype(T1()-1),N,D0,D1,D2,D3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i]-1; }
      return ret;
    }
    template <class T1, int N, class B0, class B1, class B2, class B3>
    YAKL_INLINE FSArray<decltype(T1()-1),N,B0,B1,B2,B3>
    operator--( FSArray<T1,N,B0,B1,B2,B3> const &left , int dummy ) {
      FSArray<decltype(T1()-1),N,B0,B1,B2,B3> ret;
      for (index_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i]-1; }
      return ret;
    }



  }

}


