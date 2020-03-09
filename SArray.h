
#pragma once

/*
  This is intended to be a simple, low-overhead class to do multi-dimensional arrays
  without pointer dereferencing. It supports indexing and cout only up to 3-D.

  It templates based on array dimension sizes, which conveniently allows overloaded
  functions in the TransformMatrices class.
*/

template <class T, unsigned int D0, unsigned int D1=1, unsigned int D2=1, unsigned int D3=1> class SArray {
public :

  static unsigned int constexpr totElems() { return D3*D2*D1*D0; }

  static unsigned int constexpr OFF0 = D3*D2*D1;
  static unsigned int constexpr OFF1 = D3*D2;
  static unsigned int constexpr OFF2 = D3;
  static unsigned int constexpr OFF3 = 1;

  typedef unsigned int uint;

  T mutable myData[D0*D1*D2*D3];

  YAKL_INLINE SArray() { }
  YAKL_INLINE SArray(SArray &&in) {
    for (uint i=0; i < totElems(); i++) { myData[i] = in.myData[i]; }
  }
  YAKL_INLINE SArray(SArray const &in) {
    for (uint i=0; i < totElems(); i++) { myData[i] = in.myData[i]; }
  }
  YAKL_INLINE SArray &operator=(SArray &&in) {
    for (uint i=0; i < totElems(); i++) { myData[i] = in.myData[i]; }
    return *this;
  }
  YAKL_INLINE ~SArray() { }

  YAKL_INLINE T &operator()(uint const i0) const {
    static_assert(D1==1 && D2==1 && D3==1,"ERROR: Improper number of dimensions specified in operator()");
    #ifdef ARRAY_DEBUG
      if (i0<0 || i0>D0-1) { printf("SArray i0 out of bounds (i0: %d; lb0: %d; ub0: %d",i0,0,D0-1); exit(-1); }
    #endif
    return myData[i0];
  }
  YAKL_INLINE T &operator()(uint const i0, uint const i1) const {
    static_assert(D2==1 && D3==1,"ERROR: Improper number of dimensions specified in operator()");
    #ifdef ARRAY_DEBUG
      if (i0<0 || i0>D0-1) { printf("SArray i0 out of bounds (i0: %d; lb0: %d; ub0: %d",i0,0,D0-1); exit(-1); }
      if (i1<0 || i1>D1-1) { printf("SArray i1 out of bounds (i1: %d; lb1: %d; ub1: %d",i1,0,D1-1); exit(-1); }
    #endif
    return myData[i0*OFF0 + i1];
  }
  YAKL_INLINE T &operator()(uint const i0, uint const i1, uint const i2) const {
    static_assert(D3==1,"ERROR: Improper number of dimensions specified in operator()");
    #ifdef ARRAY_DEBUG
      if (i0<0 || i0>D0-1) { printf("SArray i0 out of bounds (i0: %d; lb0: %d; ub0: %d",i0,0,D0-1); exit(-1); }
      if (i1<0 || i1>D1-1) { printf("SArray i1 out of bounds (i1: %d; lb1: %d; ub1: %d",i1,0,D1-1); exit(-1); }
      if (i2<0 || i2>D2-1) { printf("SArray i2 out of bounds (i2: %d; lb2: %d; ub2: %d",i2,0,D2-1); exit(-1); }
    #endif
    return myData[i0*OFF0 + i1*OFF1 + i2];
  }
  YAKL_INLINE T &operator()(uint const i0, uint const i1, uint const i2, uint const i3) const {
    #ifdef ARRAY_DEBUG
      if (i0<0 || i0>D0-1) { printf("SArray i0 out of bounds (i0: %d; lb0: %d; ub0: %d",i0,0,D0-1); exit(-1); }
      if (i1<0 || i1>D1-1) { printf("SArray i1 out of bounds (i1: %d; lb1: %d; ub1: %d",i1,0,D1-1); exit(-1); }
      if (i2<0 || i2>D2-1) { printf("SArray i2 out of bounds (i2: %d; lb2: %d; ub2: %d",i2,0,D2-1); exit(-1); }
      if (i3<0 || i3>D3-1) { printf("SArray i3 out of bounds (i3: %d; lb3: %d; ub3: %d",i3,0,D3-1); exit(-1); }
    #endif
    return myData[i0*OFF0 + i1*OFF1 + i2*OFF2 + i3];
  }

  YAKL_INLINE T *data() {
    return myData;
  }

  inline friend std::ostream &operator<<(std::ostream& os, SArray const &v) {
    if (D1*D2*D3 == 1) {
      for (uint i=0; i<D0; i++) {
        os << std::setw(12) << v(i) << "\n";
      }
    } else if (D2*D3 == 1) {
      for (uint j=0; j<D1; j++) {
        for (uint i=0; i<D0; i++) {
          os << std::setw(12) << v(i,j) << " ";
        }
        os << "\n";
      }
    } else {
      for (uint i=0; i<D0*D1*D2*D3; i++) {
        os << std::setw(12) << v.myData[i] << "\n";
      }
    }
    return os;
  }

};

