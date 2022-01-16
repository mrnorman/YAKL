
#pragma once

/*
  This is intended to be a simple, low-overhead class to do multi-dimensional arrays
  without pointer dereferencing. It supports indexing and cout only up to 3-D.

  It templates based on array dimension sizes, which conveniently allows overloaded
  functions in the TransformMatrices class.
*/

template <class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
class Array< CSPEC< T , D0 , D1 , D2 , D3 > , rank , memStack , styleC > {
public :

  typedef typename std::remove_cv<T>::type type;
  typedef          T value_type;
  typedef typename std::add_const<type>::type const_value_type;
  typedef typename std::remove_const<type>::type non_const_value_type;

  static unsigned constexpr totElems() { return D3*D2*D1*D0; }

  static unsigned constexpr OFF0 = D3*D2*D1;
  static unsigned constexpr OFF1 = D3*D2;
  static unsigned constexpr OFF2 = D3;
  static unsigned constexpr OFF3 = 1;

  T mutable myData[D0*D1*D2*D3];

  YAKL_INLINE Array() { }
  YAKL_INLINE Array           (Array      &&in) { for (uint i=0; i < totElems(); i++) { myData[i] = in.myData[i]; } }
  YAKL_INLINE Array           (Array const &in) { for (uint i=0; i < totElems(); i++) { myData[i] = in.myData[i]; } }
  YAKL_INLINE Array &operator=(Array      &&in) { for (uint i=0; i < totElems(); i++) { myData[i] = in.myData[i]; }; return *this; }
  YAKL_INLINE Array &operator=(Array const &in) { for (uint i=0; i < totElems(); i++) { myData[i] = in.myData[i]; }; return *this; }
  YAKL_INLINE ~Array() { }

  YAKL_INLINE T &operator()(uint const i0) const {
    static_assert(rank==1,"ERROR: Improper number of dimensions specified in operator()");
    #ifdef YAKL_DEBUG
      if (i0>D0-1) { printf("Array i0 out of bounds (i0: %d; lb0: %d; ub0: %d)\n",i0,0,D0-1); yakl_throw(""); }
    #endif
    return myData[i0];
  }
  YAKL_INLINE T &operator()(uint const i0, uint const i1) const {
    static_assert(rank==2,"ERROR: Improper number of dimensions specified in operator()");
    #ifdef YAKL_DEBUG
      if (i0>D0-1) { printf("Array i0 out of bounds (i0: %d; lb0: %d; ub0: %d)\n",i0,0,D0-1); yakl_throw(""); }
      if (i1>D1-1) { printf("Array i1 out of bounds (i1: %d; lb1: %d; ub1: %d)\n",i1,0,D1-1); yakl_throw(""); }
    #endif
    return myData[i0*OFF0 + i1];
  }
  YAKL_INLINE T &operator()(uint const i0, uint const i1, uint const i2) const {
    static_assert(rank==3,"ERROR: Improper number of dimensions specified in operator()");
    #ifdef YAKL_DEBUG
      if (i0>D0-1) { printf("Array i0 out of bounds (i0: %d; lb0: %d; ub0: %d)\n",i0,0,D0-1); yakl_throw(""); }
      if (i1>D1-1) { printf("Array i1 out of bounds (i1: %d; lb1: %d; ub1: %d)\n",i1,0,D1-1); yakl_throw(""); }
      if (i2>D2-1) { printf("Array i2 out of bounds (i2: %d; lb2: %d; ub2: %d)\n",i2,0,D2-1); yakl_throw(""); }
    #endif
    return myData[i0*OFF0 + i1*OFF1 + i2];
  }
  YAKL_INLINE T &operator()(uint const i0, uint const i1, uint const i2, uint const i3) const {
    static_assert(rank==4,"ERROR: Improper number of dimensions specified in operator()");
    #ifdef YAKL_DEBUG
      if (i0>D0-1) { printf("Array i0 out of bounds (i0: %d; lb0: %d; ub0: %d)\n",i0,0,D0-1); yakl_throw(""); }
      if (i1>D1-1) { printf("Array i1 out of bounds (i1: %d; lb1: %d; ub1: %d)\n",i1,0,D1-1); yakl_throw(""); }
      if (i2>D2-1) { printf("Array i2 out of bounds (i2: %d; lb2: %d; ub2: %d)\n",i2,0,D2-1); yakl_throw(""); }
      if (i3>D3-1) { printf("Array i3 out of bounds (i3: %d; lb3: %d; ub3: %d)\n",i3,0,D3-1); yakl_throw(""); }
    #endif
    return myData[i0*OFF0 + i1*OFF1 + i2*OFF2 + i3];
  }


  template <class TLOC , typename std::enable_if<std::is_arithmetic<TLOC>::value,int>::type = 0 >
  YAKL_INLINE void operator= (TLOC val) { for (int i=0 ; i < totElems() ; i++) { myData[i] = val; } }


  YAKL_INLINE T *data() {
    return myData;
  }


  inline friend std::ostream &operator<<(std::ostream& os, Array<CSPEC<T,D0,D1,D2,D3>,rank,memStack,styleC> const &v) {
    for (uint i=0; i<totElems(); i++) { os << std::setw(12) << v.myData[i] << "\n"; }
    os << "\n";
    return os;
  }


  template <class I> YAKL_INLINE void operator/= (I const val) {
    for (uint i=0; i < totElems(); i++) { myData[i] /= val; }
  }

  
  YAKL_INLINE Array<CSPEC<uint,rank>,1,memStack,styleC> get_dimensions() const {
    Array<CSPEC<uint,rank>,1,memStack,styleC> ret;
                     ret(0) = D0;
    if (rank >= 2) { ret(1) = D1; }
    if (rank >= 3) { ret(2) = D2; }
    if (rank >= 4) { ret(3) = D3; }
    return ret;
  }
  YAKL_INLINE Array<CSPEC<int,rank>,1,memStack,styleC> get_lbounds() const {
    Array<CSPEC<int,rank>,1,memStack,styleC> ret;
                     ret(0) = 0;
    if (rank >= 2) { ret(1) = 0; }
    if (rank >= 3) { ret(2) = 0; }
    if (rank >= 4) { ret(3) = 0; }
    return ret;
  }
  YAKL_INLINE Array<CSPEC<int,rank>,1,memStack,styleC> get_ubounds() const {
    Array<CSPEC<int,rank>,1,memStack,styleC> ret;
                     ret(0) = D0-1;
    if (rank >= 2) { ret(1) = D1-1; }
    if (rank >= 3) { ret(2) = D2-1; }
    if (rank >= 4) { ret(3) = D3-1; }
    return ret;
  }

};

