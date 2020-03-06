
#pragma once

template <class T, int rank, int myMem=memDefault, int myStyle=styleDefault> class Array;

#include "CArray.h"

#include "FArray.h"

#include "SArray.h"

#include "FSArray.h"

template <class T, int rank, int myMem, int myStyle> T minval( Array<T,rank,myMem,myStyle> &arr ) {
  ParallelMin<T,myMem> pmin(arr.totElems());
  return pmin(arr.data());
}
template <class T, int rank, int myMem, int myStyle> T minvalDevice( Array<T,rank,myMem,myStyle> &arr ) {
  T ret;
  ParallelMin<T,myMem> pmin(arr.totElems());
  pmin.deviceReduce(arr.data(),&ret);
  return ret;
}
template <class T, int rank, int myMem, int myStyle> int minloc( Array<T,rank,myMem,myStyle> &arr ) {
  T m = minval(arr);
  int loc = -999;
  parallel_for( arr.totElems() , YAKL_LAMBDA ( int i ) {
    if (arr.myData[i] == m) { loc = i; }
  });
}


