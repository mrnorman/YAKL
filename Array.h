
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
  c::parallel_for( arr.totElems() , YAKL_LAMBDA ( int i ) {
    if (arr.myData[i] == m) { loc = i; }
  });
}
template <class T, int rank, int myMem, int myStyle> int minlocDevice( Array<T,rank,myMem,myStyle> &arr ) {
  T m = minvalDevice(arr);
  int loc = -999;
  c::parallel_for( arr.totElems() , YAKL_LAMBDA ( int i ) {
    if (arr.myData[i] == m) { loc = i; }
  });
}



template <class T, int rank, int myMem, int myStyle> T maxval( Array<T,rank,myMem,myStyle> &arr ) {
  ParallelMax<T,myMem> pmax(arr.totElems());
  return pmax(arr.data());
}
template <class T, int rank, int myMem, int myStyle> T maxvalDevice( Array<T,rank,myMem,myStyle> &arr ) {
  T ret;
  ParallelMax<T,myMem> pmax(arr.totElems());
  pmax.deviceReduce(arr.data(),&ret);
  return ret;
}
template <class T, int rank, int myMem, int myStyle> int maxloc( Array<T,rank,myMem,myStyle> &arr ) {
  T m = maxval(arr);
  int loc = -999;
  c::parallel_for( arr.totElems() , YAKL_LAMBDA ( int i ) {
    if (arr.myData[i] == m) { loc = i; }
  });
}
template <class T, int rank, int myMem, int myStyle> int maxlocDevice( Array<T,rank,myMem,myStyle> &arr ) {
  T m = maxvalDevice(arr);
  int loc = -999;
  c::parallel_for( arr.totElems() , YAKL_LAMBDA ( int i ) {
    if (arr.myData[i] == m) { loc = i; }
  });
}



template <class T, int rank, int myMem, int myStyle> T sum( Array<T,rank,myMem,myStyle> &arr ) {
  ParallelSum<T,myMem> psum(arr.totElems());
  return psum(arr.data());
}
template <class T, int rank, int myMem, int myStyle> T sumDevice( Array<T,rank,myMem,myStyle> &arr ) {
  T ret;
  ParallelSum<T,myMem> psum(arr.totElems());
  psum.deviceReduce(arr.data(),&ret);
  return ret;
}



template <class F, class T, int rank, int myMem, int myStyle> bool any( Array<T,rank,myMem,myStyle> &arr , F const &f , T val ) {
  bool ret = false;
  c::parallel_for( arr.totElems() , YAKL_LAMBDA ( int i ) {
    if ( f( arr.myData[i] , val ) ) { ret = true; }
  }
  return ret;
}
template <class T, int rank, int myMem, int myStyle> bool any_less_than( Array<T,rank,myMem,myStyle> &arr , T val ) {
  return any( arr , [](T elem, T val) { return elem < val; } , val )
}


