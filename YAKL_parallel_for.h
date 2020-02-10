
#pragma once


YAKL_INLINE void storeIndices( int const ind[1] , int &i0 ) {
  i0 = ind[0];
}
YAKL_INLINE void storeIndices( int const ind[2] , int &i0 , int &i1) {
  i0 = ind[0]; i1 = ind[1];
}
YAKL_INLINE void storeIndices( int const ind[3] , int &i0 , int &i1, int &i2) {
  i0 = ind[0]; i1 = ind[1]; i2 = ind[2];
}
YAKL_INLINE void storeIndices( int const ind[4] , int &i0 , int &i1, int &i2, int &i3) {
  i0 = ind[0]; i1 = ind[1]; i2 = ind[2]; i3 = ind[3];
}
YAKL_INLINE void storeIndices( int const ind[5] , int &i0 , int &i1, int &i2, int &i3, int &i4) {
  i0 = ind[0]; i1 = ind[1]; i2 = ind[2]; i3 = ind[3]; i4 = ind[4];
}
YAKL_INLINE void storeIndices( int const ind[6] , int &i0 , int &i1, int &i2, int &i3, int &i4, int &i5) {
  i0 = ind[0]; i1 = ind[1]; i2 = ind[2]; i3 = ind[3]; i4 = ind[4]; i5 = ind[5];
}
YAKL_INLINE void storeIndices( int const ind[7] , int &i0 , int &i1, int &i2, int &i3, int &i4, int &i5, int &i6) {
  i0 = ind[0]; i1 = ind[1]; i2 = ind[2]; i3 = ind[3]; i4 = ind[4]; i5 = ind[5]; i6 = ind[6];
}
YAKL_INLINE void storeIndices( int const ind[8] , int &i0 , int &i1, int &i2, int &i3, int &i4, int &i5, int &i6, int &i7) {
  i0 = ind[0]; i1 = ind[1]; i2 = ind[2]; i3 = ind[3]; i4 = ind[4]; i5 = ind[5]; i6 = ind[6]; i7 = ind[7];
}


#include "YAKL_parallel_for_c.h"

#include "YAKL_parallel_for_fortran.h"



