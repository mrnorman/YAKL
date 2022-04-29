
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T> YAKL_INLINE T constexpr tiny(T) { return std::numeric_limits<T>::min(); }

    template <class T, int rank, int myMem, int myStyle>
    YAKL_INLINE T constexpr tiny(Array<T,rank,myMem,myStyle> const &arr) { return std::numeric_limits<T>::min(); }

    template <class T, int rank, class D0, class D1, class D2, class D3>
    YAKL_INLINE T constexpr tiny(FSArray<T,rank,D0,D1,D2,D3> const &arr) { return std::numeric_limits<T>::min(); }

    template <class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
    YAKL_INLINE T constexpr tiny(SArray<T,rank,D0,D1,D2,D3> const &arr) { return std::numeric_limits<T>::min(); }

  }
}

