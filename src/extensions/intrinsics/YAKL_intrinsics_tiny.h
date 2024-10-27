
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    template <class T> KOKKOS_INLINE_FUNCTION T constexpr tiny(T) { return std::numeric_limits<T>::min(); }

    template <class T, int rank, int myMem, int myStyle>
    KOKKOS_INLINE_FUNCTION T constexpr tiny(Array<T,rank,myMem,myStyle> const &arr) { return std::numeric_limits<T>::min(); }

    template <class T, int rank, class D0, class D1, class D2, class D3>
    KOKKOS_INLINE_FUNCTION T constexpr tiny(FSArray<T,rank,D0,D1,D2,D3> const &arr) { return std::numeric_limits<T>::min(); }

    template <class T, int rank, size_t D0, size_t D1, size_t D2, size_t D3>
    KOKKOS_INLINE_FUNCTION T constexpr tiny(SArray<T,rank,D0,D1,D2,D3> const &arr) { return std::numeric_limits<T>::min(); }

  }
}

