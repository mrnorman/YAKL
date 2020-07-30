
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



template <class T, uint COL_L, uint ROW_L, uint COL_R>
YAKL_INLINE Array< CSPEC<T,COL_R,ROW_L> , 2 , memStack , styleC >
operator* ( Array< CSPEC<T,COL_L,ROW_L> , 2 , memStack , styleC > const &left ,
            Array< CSPEC<T,COL_R,COL_L> , 2 , memStack , styleC > const &right ) {
  Array< CSPEC<T,COL_R,ROW_L> , 2 , memStack , styleC > ret;
  for (uint i=0; i < COL_R; i++) {
    for (uint j=0; j < ROW_L; j++) {
      T tmp = 0;
      for (uint k=0; k < COL_L; k++) {
        tmp += left(k,j) * right(i,k);
      }
      ret(i,j) = tmp;
    }
  }
  return ret;
}


template<class T, uint COL_L, uint ROW_L>
YAKL_INLINE Array< CSPEC<T,ROW_L> , 1 , memStack , styleC >
operator* ( Array< CSPEC<T,COL_L,ROW_L> , 2 , memStack , styleC > const &left ,
            Array< CSPEC<T,COL_L> , 1 , memStack , styleC > const &right ) {
  Array< CSPEC<T,ROW_L> , 1 , memStack , styleC > ret;
  for (uint j=0; j < ROW_L; j++) {
    T tmp = 0;
    for (uint k=0; k < COL_L; k++) {
      tmp += left(k,j) * right(k);
    }
    ret(j) = tmp;
  }
  return ret;
}


