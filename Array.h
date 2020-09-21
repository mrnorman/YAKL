
#pragma once

template <class T, int rank, int myMem=memDefault, int myStyle=styleDefault> class Array;

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


