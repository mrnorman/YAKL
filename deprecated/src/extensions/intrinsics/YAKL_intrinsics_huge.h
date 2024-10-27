
#pragma once
// Included by YAKL_intrinsics.h

__YAKL_NAMESPACE_WRAPPER_BEGIN__
namespace yakl {
  namespace intrinsics {

    template <class T> YAKL_INLINE T constexpr huge(T) { return std::numeric_limits<T>::max(); }

    template <class T, int rank, int myMem, int myStyle>
    YAKL_INLINE T constexpr huge(Array<T,rank,myMem,myStyle> const &arr) { return std::numeric_limits<T>::max(); }

    template <class T, int rank, class D0, class D1, class D2, class D3>
    YAKL_INLINE T constexpr huge(FSArray<T,rank,D0,D1,D2,D3> const &arr) { return std::numeric_limits<T>::max(); }

    template <class T, int rank, index_t D0, index_t D1, index_t D2, index_t D3>
    YAKL_INLINE T constexpr huge(SArray<T,rank,D0,D1,D2,D3> const &arr) { return std::numeric_limits<T>::max(); }

  }
}
__YAKL_NAMESPACE_WRAPPER_END__

