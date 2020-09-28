
#pragma once

template <class T, int rank, int myMem=memDefault, int myStyle=styleDefault> class Array;


class Dims {
public:
  int data[8];
  int rank;

  Dims() {rank = 0;}
  Dims(int i0) {
    data[0] = i0;
    rank = 1;
  }
  Dims(int i0, int i1) {
    data[0] = i0;
    data[1] = i1;
    rank = 2;
  }
  Dims(int i0, int i1, int i2) {
    data[0] = i0;
    data[1] = i1;
    data[2] = i2;
    rank = 3;
  }
  Dims(int i0, int i1, int i2, int i3) {
    data[0] = i0;
    data[1] = i1;
    data[2] = i2;
    data[3] = i3;
    rank = 4;
  }
  Dims(int i0, int i1, int i2, int i3, int i4) {
    data[0] = i0;
    data[1] = i1;
    data[2] = i2;
    data[3] = i3;
    data[4] = i4;
    rank = 5;
  }
  Dims(int i0, int i1, int i2, int i3, int i4, int i5) {
    data[0] = i0;
    data[1] = i1;
    data[2] = i2;
    data[3] = i3;
    data[4] = i4;
    data[5] = i5;
    rank = 6;
  }
  Dims(int i0, int i1, int i2, int i3, int i4, int i5, int i6) {
    data[0] = i0;
    data[1] = i1;
    data[2] = i2;
    data[3] = i3;
    data[4] = i4;
    data[5] = i5;
    data[6] = i6;
    rank = 7;
  }
  Dims(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7) {
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



#include "CSArray.h"
template <class T, int rank, unsigned D0, unsigned D1=1, unsigned D2=1, unsigned D3=1>
using SArray  = Array< CSPEC< T , D0 , D1 , D2 , D3 > , rank , memStack , styleC >;

#include "FSArray.h"
template <class T, int rank, class B0 , class B1=SB<1,1> , class B2=SB<1,1> , class B3=SB<1,1> >
using FSArray = Array< FSPEC< T , B0 , B1 , B2 , B3 > , rank , memStack , styleFortran >;
        
#include "CArray.h"

#include "FArray.h"



///////////////////////////////////////////////////////////
// operator* for stack arrays only
///////////////////////////////////////////////////////////
template <class T, index_t COL_L, index_t ROW_L, index_t COL_R>
YAKL_INLINE SArray<T,2,COL_R,ROW_L>
operator* ( SArray<T,2,COL_L,ROW_L> const &left ,
            SArray<T,2,COL_R,COL_L> const &right ) {
  SArray<T,2,COL_R,ROW_L> ret;
  for (index_t i=0; i < COL_R; i++) {
    for (index_t j=0; j < ROW_L; j++) {
      T tmp = 0;
      for (index_t k=0; k < COL_L; k++) {
        tmp += left(k,j) * right(i,k);
      }
      ret(i,j) = tmp;
    }
  }
  return ret;
}

template<class T, index_t COL_L, index_t ROW_L>
YAKL_INLINE SArray<T,1,ROW_L>
operator* ( SArray<T,2,COL_L,ROW_L> const &left ,
            SArray<T,1,COL_L>       const &right ) {
  SArray<T,1,ROW_L> ret;
  for (index_t j=0; j < ROW_L; j++) {
    T tmp = 0;
    for (index_t k=0; k < COL_L; k++) {
      tmp += left(k,j) * right(k);
    }
    ret(j) = tmp;
  }
  return ret;
}

template <class T, int D0_L, int D1_L, int D1_R>
YAKL_INLINE FSArray<T,2,SB<D0_L>,SB<D1_R>>
operator*( FSArray<T,2,SB<D0_L>,SB<D1_L>> const &a1 ,
           FSArray<T,2,SB<D1_L>,SB<D1_R>> const &a2 ) {
  FSArray<T,2,SB<D0_L>,SB<D1_R>> ret;
  for (int i=1; i <= D0_L; i++) {
    for (int j=1; j <= D1_R; j++) {
      T tmp = 0;
      for (int k=1; k <= D1_L; k++) {
        tmp += a1(i,k) * a2(k,j);
      }
      ret(i,j) = tmp;
    }
  }
  return ret;
}

template <class T, int D0_L, int D1_L>
YAKL_INLINE FSArray<T,1,SB<D0_L>>
operator*( FSArray<T,2,SB<D0_L>,SB<D1_L>> const &a1 ,
           FSArray<T,1,SB<D1_L>>          const &a2 ) {
  FSArray<T,1,SB<D0_L>> ret;
  for (int i=1; i <= D0_L; i++) {
    T tmp = 0;
    for (int k=1; k <= D1_L; k++) {
      tmp += a1(i,k) * a2(k);
    }
    ret(i) = tmp;
  }
  return ret;
}


