
#pragma once
// Included by YAKL_intrinsics.h

namespace yakl {
  namespace intrinsics {

    ///////////////////////////////////////////////////////////
    // Matrix multiplication routines for column-row format
    ///////////////////////////////////////////////////////////
    template <class T, size_t COL_L, size_t ROW_L, size_t COL_R>
    KOKKOS_INLINE_FUNCTION SArray<T,2,COL_R,ROW_L>
    matmul_cr ( SArray<T,2,COL_L,ROW_L> const &left ,
                SArray<T,2,COL_R,COL_L> const &right ) {
      SArray<T,2,COL_R,ROW_L> ret;
      for (size_t i=0; i < COL_R; i++) {
        for (size_t j=0; j < ROW_L; j++) {
          T tmp = 0;
          for (size_t k=0; k < COL_L; k++) {
            tmp += left(k,j) * right(i,k);
          }
          ret(i,j) = tmp;
        }
      }
      return ret;
    }

    template<class T, size_t COL_L, size_t ROW_L>
    KOKKOS_INLINE_FUNCTION SArray<T,1,ROW_L>
    matmul_cr ( SArray<T,2,COL_L,ROW_L> const &left ,
                SArray<T,1,COL_L>       const &right ) {
      SArray<T,1,ROW_L> ret;
      for (size_t j=0; j < ROW_L; j++) {
        T tmp = 0;
        for (size_t k=0; k < COL_L; k++) {
          tmp += left(k,j) * right(k);
        }
        ret(j) = tmp;
      }
      return ret;
    }

    template <class T, int COL_L, int ROW_L, int COL_R>
    KOKKOS_INLINE_FUNCTION FSArray<T,2,SB<COL_R>,SB<ROW_L>>
    matmul_cr ( FSArray<T,2,SB<COL_L>,SB<ROW_L>> const &left ,
                FSArray<T,2,SB<COL_R>,SB<COL_L>> const &right ) {
      FSArray<T,2,SB<COL_R>,SB<ROW_L>> ret;
      for (size_t i=1; i <= COL_R; i++) {
        for (size_t j=1; j <= ROW_L; j++) {
          T tmp = 0;
          for (size_t k=1; k <= COL_L; k++) {
            tmp += left(k,j) * right(i,k);
          }
          ret(i,j) = tmp;
        }
      }
      return ret;
    }

    template<class T, int COL_L, int ROW_L>
    KOKKOS_INLINE_FUNCTION FSArray<T,1,SB<ROW_L>>
    matmul_cr ( FSArray<T,2,SB<COL_L>,SB<ROW_L>> const &left ,
                FSArray<T,1,SB<COL_L>>           const &right ) {
      FSArray<T,1,SB<ROW_L>> ret;
      for (size_t j=1; j <= ROW_L; j++) {
        T tmp = 0;
        for (size_t k=1; k <= COL_L; k++) {
          tmp += left(k,j) * right(k);
        }
        ret(j) = tmp;
      }
      return ret;
    }


    ///////////////////////////////////////////////////////////
    // Matrix multiplication routines for row-column format
    ///////////////////////////////////////////////////////////
    template <class T, size_t COL_L, size_t ROW_L, size_t COL_R>
    KOKKOS_INLINE_FUNCTION SArray<T,2,ROW_L,COL_R>
    matmul_rc ( SArray<T,2,ROW_L,COL_L> const &left ,
                SArray<T,2,COL_L,COL_R> const &right ) {
      SArray<T,2,ROW_L,COL_R> ret;
      for (size_t i=0; i < COL_R; i++) {
        for (size_t j=0; j < ROW_L; j++) {
          T tmp = 0;
          for (size_t k=0; k < COL_L; k++) {
            tmp += left(j,k) * right(k,i);
          }
          ret(j,i) = tmp;
        }
      }
      return ret;
    }

    template<class T, size_t COL_L, size_t ROW_L>
    KOKKOS_INLINE_FUNCTION SArray<T,1,ROW_L>
    matmul_rc ( SArray<T,2,ROW_L,COL_L> const &left ,
                SArray<T,1,COL_L>       const &right ) {
      SArray<T,1,ROW_L> ret;
      for (size_t j=0; j < ROW_L; j++) {
        T tmp = 0;
        for (size_t k=0; k < COL_L; k++) {
          tmp += left(j,k) * right(k);
        }
        ret(j) = tmp;
      }
      return ret;
    }

    template <class T, int COL_L, int ROW_L, int COL_R>
    KOKKOS_INLINE_FUNCTION FSArray<T,2,SB<ROW_L>,SB<COL_R>>
    matmul_rc ( FSArray<T,2,SB<ROW_L>,SB<COL_L>> const &left ,
                FSArray<T,2,SB<COL_L>,SB<COL_R>> const &right ) {
      FSArray<T,2,SB<ROW_L>,SB<COL_R>> ret;
      for (size_t i=1; i <= COL_R; i++) {
        for (size_t j=1; j <= ROW_L; j++) {
          T tmp = 0;
          for (size_t k=1; k <= COL_L; k++) {
            tmp += left(j,k) * right(k,i);
          }
          ret(j,i) = tmp;
        }
      }
      return ret;
    }

    template<class T, int COL_L, int ROW_L>
    KOKKOS_INLINE_FUNCTION FSArray<T,1,SB<ROW_L>>
    matmul_rc ( FSArray<T,2,SB<ROW_L>,SB<COL_L>> const &left ,
                FSArray<T,1,SB<COL_L>>           const &right ) {
      FSArray<T,1,SB<ROW_L>> ret;
      for (size_t j=1; j <= ROW_L; j++) {
        T tmp = 0;
        for (size_t k=1; k <= COL_L; k++) {
          tmp += left(j,k) * right(k);
        }
        ret(j) = tmp;
      }
      return ret;
    }

  }
}
