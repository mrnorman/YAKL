
#pragma once

/*
  This is intended to be a simple, low-overhead class to do multi-dimensional arrays
  without pointer dereferencing. It supports indexing and cout only up to 4-D.
*/

template <class T, typename B0, typename B1=SBnd<1,1>, typename B2=SBnd<1,1>, typename B3=SBnd<1,1>> class FSArray {
public :
  static unsigned constexpr D0 = B0::u() - B0::l() + 1;
  static unsigned constexpr D1 = B1::u() - B1::l() + 1;
  static unsigned constexpr D2 = B2::u() - B2::l() + 1;
  static unsigned constexpr D3 = B3::u() - B3::l() + 1;

  static unsigned constexpr totElems() { return D0*D1*D2*D3; }

  static unsigned constexpr OFF0 = 1;
  static unsigned constexpr OFF1 = D0;
  static unsigned constexpr OFF2 = D0*D1;
  static unsigned constexpr OFF3 = D0*D1*D2;

  T mutable myData[D0*D1*D2*D3];

  YAKL_INLINE FSArray() { }
  YAKL_INLINE FSArray(FSArray &&in) {
    for (int i=0; i < totElems(); i++) { myData[i] = in.myData[i]; }
  }
  YAKL_INLINE FSArray(FSArray const &in) {
    for (int i=0; i < totElems(); i++) { myData[i] = in.myData[i]; }
  }
  YAKL_INLINE FSArray &operator=(FSArray &&in) {
    for (int i=0; i < totElems(); i++) { myData[i] = in.myData[i]; }
    return *this;
  }
  YAKL_INLINE ~FSArray() { }

  YAKL_INLINE T &operator()(int const i0) const {
    static_assert(D1==1 && D2==1 && D3==1,"ERROR: Improper number of dimensions specified in operator()");
    #ifdef ARRAY_DEBUG
      if (i0<B0::l() || i0>B0::u()) { printf("FSArray i0 out of bounds (i0: %d; lb0: %d; ub0: %d",i0,B0::l(),B0::u()); exit(-1); }
    #endif
    return myData[i0-B0::l()];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1) const {
    static_assert(D2==1 && D3==1,"ERROR: Improper number of dimensions specified in operator()");
    #ifdef ARRAY_DEBUG
      if (i0<B0::l() || i0>B0::u()) { printf("FSArray i0 out of bounds (i0: %d; lb0: %d; ub0: %d",i0,B0::l(),B0::u()); exit(-1); }
      if (i1<B1::l() || i1>B1::u()) { printf("FSArray i1 out of bounds (i1: %d; lb1: %d; ub1: %d",i1,B1::l(),B1::u()); exit(-1); }
    #endif
    return myData[(i1-B1::l())*OFF1 + i0-B0::l()];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1, int const i2) const {
    static_assert(D3==1,"ERROR: Improper number of dimensions specified in operator()");
    #ifdef ARRAY_DEBUG
      if (i0<B0::l() || i0>B0::u()) { printf("FSArray i0 out of bounds (i0: %d; lb0: %d; ub0: %d",i0,B0::l(),B0::u()); exit(-1); }
      if (i1<B1::l() || i1>B1::u()) { printf("FSArray i1 out of bounds (i1: %d; lb1: %d; ub1: %d",i1,B1::l(),B1::u()); exit(-1); }
      if (i2<B2::l() || i2>B2::u()) { printf("FSArray i2 out of bounds (i2: %d; lb2: %d; ub2: %d",i2,B2::l(),B2::u()); exit(-1); }
    #endif
    return myData[(i2-B2::l())*OFF2 + (i1-B1::l())*OFF1 + i0-B0::l()];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1, int const i2, int const i3) const {
    #ifdef ARRAY_DEBUG
      if (i0<B0::l() || i0>B0::u()) { printf("FSArray i0 out of bounds (i0: %d; lb0: %d; ub0: %d",i0,B0::l(),B0::u()); exit(-1); }
      if (i1<B1::l() || i1>B1::u()) { printf("FSArray i1 out of bounds (i1: %d; lb1: %d; ub1: %d",i1,B1::l(),B1::u()); exit(-1); }
      if (i2<B2::l() || i2>B2::u()) { printf("FSArray i2 out of bounds (i2: %d; lb2: %d; ub2: %d",i2,B2::l(),B2::u()); exit(-1); }
      if (i3<B3::l() || i3>B3::u()) { printf("FSArray i3 out of bounds (i3: %d; lb3: %d; ub3: %d",i3,B3::l(),B3::u()); exit(-1); }
    #endif
    return myData[(i3-B3::l())*OFF3 + (i2-B2::l())*OFF2 + (i1-B1::l())*OFF1 + i0-B0::l()];
  }

  YAKL_INLINE T *data() {
    return myData;
  }

  inline friend std::ostream &operator<<(std::ostream& os, FSArray const &v) {
    if (D1*D2*D3 == 1) {
      for (int i=B0::l(); i<=B0::u(); i++) {
        os << std::setw(12) << v(i) << "\n";
      }
    } else if (D2*D3 == 1) {
      for (int j=B1::l(); j<B1::u(); j++) {
        for (int i=B0::l(); i<B0::u(); i++) {
          os << std::setw(12) << v(i,j) << " ";
        }
        os << "\n";
      }
    } else {
      for (int i=0; i<D0*D1*D2*D3; i++) {
        os << std::setw(12) << v.myData[i] << "\n";
      }
    }
    return os;
  }

};

