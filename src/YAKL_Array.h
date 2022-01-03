
#pragma once

template <class T, int rank, int myMem=memDefault, int myStyle=styleDefault> class Array;


class Dims {
public:
  int data[8];
  int rank;

  YAKL_INLINE Dims() {rank = 0;}
  YAKL_INLINE Dims(int i0) {
    data[0] = i0;
    rank = 1;
  }
  YAKL_INLINE Dims(int i0, int i1) {
    data[0] = i0;
    data[1] = i1;
    rank = 2;
  }
  YAKL_INLINE Dims(int i0, int i1, int i2) {
    data[0] = i0;
    data[1] = i1;
    data[2] = i2;
    rank = 3;
  }
  YAKL_INLINE Dims(int i0, int i1, int i2, int i3) {
    data[0] = i0;
    data[1] = i1;
    data[2] = i2;
    data[3] = i3;
    rank = 4;
  }
  YAKL_INLINE Dims(int i0, int i1, int i2, int i3, int i4) {
    data[0] = i0;
    data[1] = i1;
    data[2] = i2;
    data[3] = i3;
    data[4] = i4;
    rank = 5;
  }
  YAKL_INLINE Dims(int i0, int i1, int i2, int i3, int i4, int i5) {
    data[0] = i0;
    data[1] = i1;
    data[2] = i2;
    data[3] = i3;
    data[4] = i4;
    data[5] = i5;
    rank = 6;
  }
  YAKL_INLINE Dims(int i0, int i1, int i2, int i3, int i4, int i5, int i6) {
    data[0] = i0;
    data[1] = i1;
    data[2] = i2;
    data[3] = i3;
    data[4] = i4;
    data[5] = i5;
    data[6] = i6;
    rank = 7;
  }
  YAKL_INLINE Dims(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7) {
    data[0] = i0;
    data[1] = i1;
    data[2] = i2;
    data[3] = i3;
    data[4] = i4;
    data[5] = i5;
    data[6] = i6;
    data[7] = i7;
    rank = 8;
  }

  YAKL_INLINE int size() const {
    return rank;
  }
};



// Dynamic (runtime) Array Bounds
class Bnd {
public:
  int l, u;
  YAKL_INLINE Bnd(                  ) { l = 1   ; u = 1   ; }
  YAKL_INLINE Bnd(          int u_in) { l = 1   ; u = u_in; }
  YAKL_INLINE Bnd(int l_in, int u_in) { l = l_in; u = u_in; }
};



class Bnds {
public:
  int l[8];
  int u[8];
  int rank;

  YAKL_INLINE Bnds() {rank = 0;}
  YAKL_INLINE Bnds(Bnd b0) {
    l[0] = b0.l;

    u[0] = b0.u;

    rank = 1;
  }
  YAKL_INLINE Bnds(Bnd b0, Bnd b1) {
    l[0] = b0.l;
    l[1] = b1.l;

    u[0] = b0.u;
    u[1] = b1.u;

    rank = 2;
  }
  YAKL_INLINE Bnds(Bnd b0, Bnd b1, Bnd b2) {
    l[0] = b0.l;
    l[1] = b1.l;
    l[2] = b2.l;

    u[0] = b0.u;
    u[1] = b1.u;
    u[2] = b2.u;

    rank = 3;
  }
  YAKL_INLINE Bnds(Bnd b0, Bnd b1, Bnd b2, Bnd b3) {
    l[0] = b0.l;
    l[1] = b1.l;
    l[2] = b2.l;
    l[3] = b3.l;

    u[0] = b0.u;
    u[1] = b1.u;
    u[2] = b2.u;
    u[3] = b3.u;

    rank = 4;
  }
  YAKL_INLINE Bnds(Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4) {
    l[0] = b0.l;
    l[1] = b1.l;
    l[2] = b2.l;
    l[3] = b3.l;
    l[4] = b4.l;

    u[0] = b0.u;
    u[1] = b1.u;
    u[2] = b2.u;
    u[3] = b3.u;
    u[4] = b4.u;

    rank = 5;
  }
  YAKL_INLINE Bnds(Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5) {
    l[0] = b0.l;
    l[1] = b1.l;
    l[2] = b2.l;
    l[3] = b3.l;
    l[4] = b4.l;
    l[5] = b5.l;

    u[0] = b0.u;
    u[1] = b1.u;
    u[2] = b2.u;
    u[3] = b3.u;
    u[4] = b4.u;
    u[5] = b5.u;

    rank = 6;
  }
  YAKL_INLINE Bnds(Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5, Bnd b6) {
    l[0] = b0.l;
    l[1] = b1.l;
    l[2] = b2.l;
    l[3] = b3.l;
    l[4] = b4.l;
    l[5] = b5.l;
    l[6] = b6.l;

    u[0] = b0.u;
    u[1] = b1.u;
    u[2] = b2.u;
    u[3] = b3.u;
    u[4] = b4.u;
    u[5] = b5.u;
    u[6] = b6.u;

    rank = 7;
  }
  YAKL_INLINE Bnds(Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5, Bnd b6, Bnd b7) {
    l[0] = b0.l;
    l[1] = b1.l;
    l[2] = b2.l;
    l[3] = b3.l;
    l[4] = b4.l;
    l[5] = b5.l;
    l[6] = b6.l;
    l[7] = b7.l;

    u[0] = b0.u;
    u[1] = b1.u;
    u[2] = b2.u;
    u[3] = b3.u;
    u[4] = b4.u;
    u[5] = b5.u;
    u[6] = b6.u;
    u[7] = b7.u;

    rank = 8;
  }

  int size() const {
    return rank;
  }
};



// [S]tatic (compile-time) Array [B]ounds (templated)
// It's only used for Fortran, so it takes on Fortran defaults
// with lower bound default to 1
template <int L, int U=-999> class SB {
public:
  SB() = delete;
};



// Fortran list of static bounds
template <class T, class B0, class B1=SB<1,1>, class B2=SB<1,1>, class B3=SB<1,1>> class FSPEC {
public:
  FSPEC() = delete;
};



// C list of static dimension sizes
template <class T, unsigned D0, unsigned D1=1, unsigned D2=1, unsigned D3=1> class CSPEC {
public:
  CSPEC() = delete;
};



#include "YAKL_CSArray.h"
template <class T, int rank, unsigned D0, unsigned D1=1, unsigned D2=1, unsigned D3=1>
using SArray  = Array< CSPEC< T , D0 , D1 , D2 , D3 > , rank , memStack , styleC >;

#include "YAKL_FSArray.h"
template <class T, int rank, class B0 , class B1=SB<1,1> , class B2=SB<1,1> , class B3=SB<1,1> >
using FSArray = Array< FSPEC< T , B0 , B1 , B2 , B3 > , rank , memStack , styleFortran >;
        
#include "YAKL_ArrayBase.h"

#include "YAKL_CArrayBase.h"
#include "YAKL_CArray.h"

#include "YAKL_FArrayBase.h"
#include "YAKL_FArray.h"






