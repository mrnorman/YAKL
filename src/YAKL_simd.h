/**
 * @file
 * YAKL Pack routines to encourage SIMD vectorization
 */

#pragma once
// Included by YAKL.h

#if defined(__GNUG__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && !defined(__NVCOMPILER)
# define GET_SIMD_PRAGMA() _Pragma("GCC ivdep")
#elif defined(__clang__) && !defined(__INTEL_COMPILER) && !defined(__NVCOMPILER)
# define GET_SIMD_PRAGMA() _Pragma("clang loop vectorize(enable)")
#elif defined(__INTEL_COMPILER) || defined(__NVCOMPILER)
# define GET_SIMD_PRAGMA() _Pragma("ivdep")
#else
# define GET_SIMD_PRAGMA()
#endif

namespace yakl {

  // Packs can be used by the user in order to encourage compilers to SIMD vectorize in cases where the compiler
  // is having trouble determining when vector instructions can be generated. For recent versions of compilers,
  // this is not very necessary.
  // 
  // Pack objects are fairly bare bones in terms of functionality

  template <class T, unsigned int N>
  class Pack {
  public:
    typedef unsigned int uint;
    T myData[N];

    YAKL_INLINE T & operator() (uint i) {
      #ifdef YAKL_DEBUG
        if (i >= N) { yakl_throw("Pack index out of bounds"); }
      #endif
      return myData[i];
    }

    YAKL_INLINE T operator() (uint i) const {
      #ifdef YAKL_DEBUG
        if (i >= N) { yakl_throw("Pack index out of bounds"); }
      #endif
      return myData[i];
    }

    YAKL_INLINE static int constexpr get_pack_size() { return N; }

    //////////////////////////////////////
    // SELF OPERATORS WITH SCALAR VALUES
    //////////////////////////////////////
    template <class TLOC , typename std::enable_if<std::is_arithmetic<TLOC>::value,bool>::type = false >
    YAKL_INLINE Pack<T,N> & operator= (TLOC rhs) {
      GET_SIMD_PRAGMA()
      for (int i=0 ; i < N ; i++) { (*this)(i) = rhs; }
      return *this;
    }

    template <class TLOC , typename std::enable_if<std::is_arithmetic<TLOC>::value,bool>::type = false >
    YAKL_INLINE Pack<T,N> & operator+= (TLOC rhs) {
      GET_SIMD_PRAGMA()
      for (uint i=0; i < N; i++) { (*this)(i) += rhs; }
      return *this;
    }

    template <class TLOC , typename std::enable_if<std::is_arithmetic<TLOC>::value,bool>::type = false >
    YAKL_INLINE Pack<T,N> & operator-= (TLOC rhs) {
      GET_SIMD_PRAGMA()
      for (uint i=0; i < N; i++) { (*this)(i) -= rhs; }
      return *this;
    }

    template <class TLOC , typename std::enable_if<std::is_arithmetic<TLOC>::value,bool>::type = false >
    YAKL_INLINE Pack<T,N> & operator*= (TLOC rhs) {
      GET_SIMD_PRAGMA()
      for (uint i=0; i < N; i++) { (*this)(i) *= rhs; }
      return *this;
    }

    template <class TLOC , typename std::enable_if<std::is_arithmetic<TLOC>::value,bool>::type = false >
    YAKL_INLINE Pack<T,N> & operator/= (TLOC rhs) {
      GET_SIMD_PRAGMA()
      for (uint i=0; i < N; i++) { (*this)(i) /= rhs; }
      return *this;
    }

    //////////////////////////////////////
    // SELF OPERATORS WITH PACKS
    //////////////////////////////////////
    template <class TLOC>
    YAKL_INLINE Pack<T,N> & operator+= (Pack<TLOC,N> rhs) {
      GET_SIMD_PRAGMA()
      for (uint i=0; i < N; i++) { (*this)(i) += rhs(i); }
      return *this;
    }

    template <class TLOC>
    YAKL_INLINE Pack<T,N> & operator-= (Pack<TLOC,N> rhs) {
      GET_SIMD_PRAGMA()
      for (uint i=0; i < N; i++) { (*this)(i) -= rhs(i); }
      return *this;
    }

    template <class TLOC>
    YAKL_INLINE Pack<T,N> & operator*= (Pack<TLOC,N> rhs) {
      GET_SIMD_PRAGMA()
      for (uint i=0; i < N; i++) { (*this)(i) *= rhs(i); }
      return *this;
    }

    template <class TLOC>
    YAKL_INLINE Pack<T,N> & operator/= (Pack<TLOC,N> rhs) {
      GET_SIMD_PRAGMA()
      for (uint i=0; i < N; i++) { (*this)(i) /= rhs(i); }
      return *this;
    }


    inline friend std::ostream &operator<<(std::ostream& os, Pack<T,N> const &v) {
      for (uint i=0; i<N; i++) { os << std::setw(12) << v(i) << "  "; }
      os << "\n";
      return os;
    }
  };


  /**
   * @brief This object must be passed to yakl::iterate_over_pack to inform that routine what the simd vector length is
   *        as well as whether to apply a SIMD pragma to the loop or not.
   */
  template <unsigned int N, bool SIMD=false> struct PackIterConfig {};



  /**
   * @brief Perform a loop over the vector length size specified by the config parameter. If the config parameter also
   *        specifies that the SIMD template parameter is true, then apply SIMD pragmas.
   *        IMPORTANT: For the functor passed to this routine, please use [&] syntax, *not* YAKL_LAMBDA
   * @param f      The functor object to execute inside the loop.
   * @param config yakl::PackIterConfig object with two template parameters: (1) the vector length (number of elements
   *               to loop over; and (2) a bool SIMD parameter to tell this routine whether or not it should apply a
   *               SIMD pragma.
   */
  template <class F, unsigned int N, bool SIMD=false>
  YAKL_INLINE void iterate_over_pack( F const &f , PackIterConfig<N,SIMD> config ) {
    if constexpr (SIMD) {
      GET_SIMD_PRAGMA()
      for (int i=0 ; i < N ; i++) { f(i); }
    } else {
      for (int i=0 ; i < N ; i++) { f(i); }
    }
  }


  //////////////////////////////////////////////////////////////
  // OPERATIONS WITH SCALARS
  //////////////////////////////////////////////////////////////
  template <class T, unsigned int N, class TLOC , typename std::enable_if<std::is_arithmetic<TLOC>::value,bool>::type = false >
  YAKL_INLINE Pack<T,N> operator+ (Pack<T,N> lhs , TLOC val) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (uint i=0; i < N; i++) { ret(i) = lhs(i) + val; }
    return ret;
  }
  template <class T, unsigned int N, class TLOC , typename std::enable_if<std::is_arithmetic<TLOC>::value,bool>::type = false >
  YAKL_INLINE Pack<T,N> operator+ (TLOC val , Pack<T,N> rhs) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (uint i=0; i < N; i++) { ret(i) = val + rhs(i); }
    return ret;
  }

  template <class T, unsigned int N, class TLOC , typename std::enable_if<std::is_arithmetic<TLOC>::value,bool>::type = false >
  YAKL_INLINE Pack<T,N> operator- (Pack<T,N> lhs , TLOC val) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (uint i=0; i < N; i++) { ret(i) = lhs(i) - val; }
    return ret;
  }
  template <class T, unsigned int N, class TLOC , typename std::enable_if<std::is_arithmetic<TLOC>::value,bool>::type = false >
  YAKL_INLINE Pack<T,N> operator- (TLOC val , Pack<T,N> rhs) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (uint i=0; i < N; i++) { ret(i) = val - rhs(i); }
    return ret;
  }

  template <class T, unsigned int N, class TLOC , typename std::enable_if<std::is_arithmetic<TLOC>::value,bool>::type = false >
  YAKL_INLINE Pack<T,N> operator* (Pack<T,N> lhs , TLOC val) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (uint i=0; i < N; i++) { ret(i) = lhs(i) * val; }
    return ret;
  }
  template <class T, unsigned int N, class TLOC , typename std::enable_if<std::is_arithmetic<TLOC>::value,bool>::type = false >
  YAKL_INLINE Pack<T,N> operator* (TLOC val , Pack<T,N> rhs) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (uint i=0; i < N; i++) { ret(i) = val * rhs(i); }
    return ret;
  }

  template <class T, unsigned int N, class TLOC , typename std::enable_if<std::is_arithmetic<TLOC>::value,bool>::type = false >
  YAKL_INLINE Pack<T,N> operator/ (Pack<T,N> lhs , TLOC val) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (uint i=0; i < N; i++) { ret(i) = lhs(i) / val; }
    return ret;
  }
  template <class T, unsigned int N, class TLOC , typename std::enable_if<std::is_arithmetic<TLOC>::value,bool>::type = false >
  YAKL_INLINE Pack<T,N> operator/ (TLOC val , Pack<T,N> rhs) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (uint i=0; i < N; i++) { ret(i) = val / rhs(i); }
    return ret;
  }

  template <class T, unsigned int N, class TLOC , typename std::enable_if<std::is_arithmetic<TLOC>::value,bool>::type = false >
  YAKL_INLINE Pack<T,N> pow(Pack<T,N> lhs , TLOC val) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (uint i=0; i < N; i++) { ret(i) = pow( lhs(i) , val ); }
    return ret;
  }


  //////////////////////////////////////////////////////////////
  // UNARY OPERATORS
  //////////////////////////////////////////////////////////////
  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> operator- ( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = -( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> sqrt( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = sqrt( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> abs( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = abs( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> exp( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = exp( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> log( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = log( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> log10( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = log10( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> cos( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = cos( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> sin( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = sin( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> tan( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = tan( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> acos( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = acos( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> asign( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = asign( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> atan( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = atan( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> ceil( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = ceil( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> floor( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = floor( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> round( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = round( a(i) ); }
    return ret;
  }



  //////////////////////////////////////////////////////////////
  // BINARY OPERATORS
  //////////////////////////////////////////////////////////////
  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> operator+( Pack<T,N> a , Pack<T,N> b) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) {
      ret(i) = a(i) + b(i);
    }
    return ret;
  }


  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> operator-( Pack<T,N> a , Pack<T,N> b) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) {
      ret(i) = a(i) - b(i);
    }
    return ret;
  }


  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> operator*( Pack<T,N> a , Pack<T,N> b) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) {
      ret(i) = a(i) * b(i);
    }
    return ret;
  }


  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> operator/( Pack<T,N> a , Pack<T,N> b) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) {
      ret(i) = a(i) / b(i);
    }
    return ret;
  }


  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> pow( Pack<T,N> a , Pack<T,N> b) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) {
      ret(i) = pow( a(i) , b(i) );
    }
    return ret;
  }

}


