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

/** @brief Holds YAKL's Pack class and operators to encourage SIMD vectorization */
namespace simd {


  /** @brief The Pack class performs encourages vectorization by performing operations Packs of known size.
    *
    * @param T Data type of the Pack
    * @param N Number of elements in the Pack object
    * 
    * Packs can be used by the user in order to encourage compilers to SIMD vectorize in cases where the compiler
    * is having trouble determining when vector instructions can be generated. 
    * 
    * For straighforward mathematical operations, simply apply that operation to the Pack object(s). 
    * 
    * Whenever different behavior is needed for different members of the Pack object (e.g., if-statement),
    * the user must iterate explicitly over the pack with yakl::iterate_over_pack.
    */
  template <class T, unsigned int N>
  class Pack {
  public:
    /** @private */
    T myData[N];

    /** @brief Returns a modifiable reference to the data at the requested index */
    YAKL_INLINE T & operator() (uint i) {
      #ifdef YAKL_DEBUG
        if (i >= N) { yakl_throw("Pack index out of bounds"); }
      #endif
      return myData[i];
    }

    /** @brief Returns a non-modifiable value to the data at the requested index */
    YAKL_INLINE T operator() (uint i) const {
      #ifdef YAKL_DEBUG
        if (i >= N) { yakl_throw("Pack index out of bounds"); }
      #endif
      return myData[i];
    }

    /** @brief Returns the number of elements in the Pack object */
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


    /** @brief Print out the Pack object values to stdout. */
    inline friend std::ostream &operator<<(std::ostream& os, Pack<T,N> const &v) {
      for (uint i=0; i<N; i++) { os << std::setw(12) << v(i) << "  "; }
      os << "\n";
      return os;
    }
  };


  /**
   * @brief Informs iterate_over_pack of the Pack size and whether to apply a SIMD pragma.
   * @details Non-parallelizeable operations should **not** specify `SIMD = true`
   * @param N    Number of elements in the Pack(s) being used inside iterate_over_pack
   * @param SIMD Whether the functor passed to iterate_over_pack is parallelizeable or not
   */
  template <unsigned int N, bool SIMD=false> struct PackIterConfig {};



  /**
   * @brief Perform a loop over the number of elements specified by the PackIterConfig object.
   * 
   * If the config parameter also specifies that the SIMD template parameter is true, then apply SIMD pragmas.
   * IMPORTANT: For the functor passed to this routine, please use **[&]** syntax, not YAKL_LAMBDA
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
    for (uint i=0; i < N; i++) { ret(i) = std::pow( lhs(i) , val ); }
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
    for (int i=0; i < N; i++) { ret(i) = std::sqrt( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> abs( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = std::abs( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> exp( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = std::exp( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> log( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = std::log( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> log10( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = std::log10( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> cos( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = std::cos( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> sin( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = std::sin( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> tan( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = std::tan( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> acos( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = std::acos( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> asin( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = std::asin( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> atan( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = std::atan( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> ceil( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = std::ceil( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> floor( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = std::floor( a(i) ); }
    return ret;
  }

  template <class T, unsigned int N>
  YAKL_INLINE Pack<T,N> round( Pack<T,N> a ) {
    Pack<T,N> ret;
    GET_SIMD_PRAGMA()
    for (int i=0; i < N; i++) { ret(i) = std::round( a(i) ); }
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
      ret(i) = std::pow( a(i) , b(i) );
    }
    return ret;
  }

}

}


