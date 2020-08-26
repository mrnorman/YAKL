
#pragma once

/*
  This is intended to be a simple, low-overhead class to do multi-dimensional arrays
  without pointer dereferencing. It supports indexing and cout only up to 4-D.
*/

template <class T, int rank, int L0_IN, int U0_IN, int L1_IN, int U1_IN, int L2_IN, int U2_IN, int L3_IN, int U3_IN>
class Array< FSPEC< T , SB<L0_IN,U0_IN> , SB<L1_IN,U1_IN> , SB<L2_IN,U2_IN> , SB<L3_IN,U3_IN> > , rank , memStack , styleFortran > {
public :
  static int constexpr U0 = U0_IN == -999 ? L0_IN : U0_IN;
  static int constexpr L0 = U0_IN == -999 ? 1     : L0_IN;
  static int constexpr U1 = U1_IN == -999 ? L1_IN : U1_IN;
  static int constexpr L1 = U1_IN == -999 ? 1     : L1_IN;
  static int constexpr U2 = U2_IN == -999 ? L2_IN : U2_IN;
  static int constexpr L2 = U2_IN == -999 ? 1     : L2_IN;
  static int constexpr U3 = U3_IN == -999 ? L3_IN : U3_IN;
  static int constexpr L3 = U3_IN == -999 ? 1     : L3_IN;

  static unsigned constexpr D0 =             U0 - L0 + 1;
  static unsigned constexpr D1 = rank >= 1 ? U1 - L1 + 1 : 1;
  static unsigned constexpr D2 = rank >= 1 ? U2 - L2 + 1 : 1;
  static unsigned constexpr D3 = rank >= 1 ? U3 - L3 + 1 : 1;

  static unsigned constexpr totElems() { return D0*D1*D2*D3; }

  static unsigned constexpr OFF0 = 1;
  static unsigned constexpr OFF1 = D0;
  static unsigned constexpr OFF2 = D0*D1;
  static unsigned constexpr OFF3 = D0*D1*D2;

  T mutable myData[D0*D1*D2*D3];

  YAKL_INLINE Array() {}
  YAKL_INLINE Array           (Array      &&in) { for (uint i=0; i < totElems(); i++) { myData[i] = in.myData[i]; } }
  YAKL_INLINE Array           (Array const &in) { for (uint i=0; i < totElems(); i++) { myData[i] = in.myData[i]; } }
  YAKL_INLINE Array &operator=(Array      &&in) { for (uint i=0; i < totElems(); i++) { myData[i] = in.myData[i]; }; return *this; }
  YAKL_INLINE Array &operator=(Array const &in) { for (uint i=0; i < totElems(); i++) { myData[i] = in.myData[i]; }; return *this; }
  YAKL_INLINE ~Array() { }

  YAKL_INLINE T &operator()(int const i0) const {
    static_assert(rank==1,"ERROR: Improper number of dimensions specified in operator()");
    #ifdef YAKL_DEBUG
      if (i0<L0 || i0>U0) { printf("Array i0 out of bounds (i0: %d; lb0: %d; ub0: %d",i0,L0,U0); throw ""; }
    #endif
    return myData[i0-L0];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1) const {
    static_assert(rank==2,"ERROR: Improper number of dimensions specified in operator()");
    #ifdef YAKL_DEBUG
      if (i0<L0 || i0>U0) { printf("Array i0 out of bounds (i0: %d; lb0: %d; ub0: %d",i0,L0,U0); throw ""; }
      if (i1<L1 || i1>U1) { printf("Array i1 out of bounds (i1: %d; lb1: %d; ub1: %d",i1,L1,U1); throw ""; }
    #endif
    return myData[(i1-L1)*OFF1 + i0-L0];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1, int const i2) const {
    static_assert(rank==3,"ERROR: Improper number of dimensions specified in operator()");
    #ifdef YAKL_DEBUG
      if (i0<L0 || i0>U0) { printf("Array i0 out of bounds (i0: %d; lb0: %d; ub0: %d",i0,L0,U0); throw ""; }
      if (i1<L1 || i1>U1) { printf("Array i1 out of bounds (i1: %d; lb1: %d; ub1: %d",i1,L1,U1); throw ""; }
      if (i2<L2 || i2>U2) { printf("Array i2 out of bounds (i2: %d; lb2: %d; ub2: %d",i2,L2,U2); throw ""; }
    #endif
    return myData[(i2-L2)*OFF2 + (i1-L1)*OFF1 + i0-L0];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1, int const i2, int const i3) const {
    static_assert(rank==4,"ERROR: Improper number of dimensions specified in operator()");
    #ifdef YAKL_DEBUG
      if (i0<L0 || i0>U0) { printf("Array i0 out of bounds (i0: %d; lb0: %d; ub0: %d",i0,L0,U0); throw ""; }
      if (i1<L1 || i1>U1) { printf("Array i1 out of bounds (i1: %d; lb1: %d; ub1: %d",i1,L1,U1); throw ""; }
      if (i2<L2 || i2>U2) { printf("Array i2 out of bounds (i2: %d; lb2: %d; ub2: %d",i2,L2,U2); throw ""; }
      if (i3<L3 || i3>U3) { printf("Array i3 out of bounds (i3: %d; lb3: %d; ub3: %d",i3,L3,U3); throw ""; }
    #endif
    return myData[(i3-L3)*OFF3 + (i2-L2)*OFF2 + (i1-L1)*OFF1 + i0-L0];
  }

  YAKL_INLINE T *data() {
    return myData;
  }


  inline friend std::ostream &operator<<(std::ostream& os, Array const &v) {
    for (int i=0; i<totElems(); i++) { os << std::setw(12) << v.myData[i] << "\n"; }
    os << "\n";
    return os;
  }


  template <int D1_R , int myrank=rank , typename std::enable_if<myrank==2,bool>::type = false >
  YAKL_INLINE Array< FSPEC<T,SB<D0>,SB<D1_R>> , 2 , memStack , styleFortran >
  operator* ( Array< FSPEC<T,SB<D1>,SB<D1_R>> , 2 , memStack , styleFortran > const &rhs ) {
    Array< FSPEC<T,SB<D0>,SB<D1_R>> , 2 , memStack , styleFortran > ret;
    for (int i=1; i <= D0; i++) {
      for (int j=1; j <= D1_R; j++) {
        T tmp = 0;
        for (int k=1; k <= D1; k++) {
          tmp += (*this)(i,k) * rhs(k,j);
        }
        ret(i,j) = tmp;
      }
    }
    return ret;
  }


  template < int myrank=rank , typename std::enable_if<myrank==2,bool>::type = false >
  YAKL_INLINE Array< FSPEC<T,SB<D0>> , 1 , memStack , styleFortran > operator* ( Array< FSPEC<T,SB<D1>> , 1 , memStack , styleFortran > const &rhs ) {
    Array< FSPEC<T,SB<D0>> , 1 , memStack , styleFortran > ret;
    for (int i=1; i <= D0; i++) {
      T tmp = 0;
      for (int k=1; k <= D1; k++) {
        tmp += (*this)(i,k) * rhs(k);
      }
      ret(i) = tmp;
    }
    return ret;
  }

  
  YAKL_INLINE Array<FSPEC<int,SB<rank>>,1,memStack,styleFortran> get_dimensions() const {
    Array<FSPEC<int,SB<rank>>,1,memStack,styleFortran> ret;
                     ret(1) = D0;
    if (rank >= 2) { ret(2) = D1; }
    if (rank >= 3) { ret(3) = D2; }
    if (rank >= 4) { ret(4) = D3; }
    return ret;
  }
  YAKL_INLINE Array<FSPEC<int,SB<rank>>,1,memStack,styleFortran> get_lbounds() const {
    Array<FSPEC<int,SB<rank>>,1,memStack,styleFortran> ret;
                     ret(1) = L0;
    if (rank >= 2) { ret(2) = L1; }
    if (rank >= 3) { ret(3) = L2; }
    if (rank >= 4) { ret(4) = L3; }
    return ret;
  }
  YAKL_INLINE Array<FSPEC<int,SB<rank>>,1,memStack,styleFortran> get_ubounds() const {
    Array<FSPEC<int,SB<rank>>,1,memStack,styleFortran> ret;
                     ret(1) = U0;
    if (rank >= 2) { ret(2) = U1; }
    if (rank >= 3) { ret(3) = U2; }
    if (rank >= 4) { ret(4) = U3; }
    return ret;
  }

};

