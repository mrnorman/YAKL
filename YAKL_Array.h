
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



// Dynamic (runtime) Array Bounds
class Bnd {
public:
  int l, u;
  Bnd(                  ) { l = 1   ; u = 1   ; }
  Bnd(          int u_in) { l = 1   ; u = u_in; }
  Bnd(int l_in, int u_in) { l = l_in; u = u_in; }
};



class Bnds {
public:
  int l[8];
  int u[8];
  int rank;

  Bnds() {rank = 0;}
  Bnds(Bnd b0) {
    l[0] = b0.l;

    u[0] = b0.u;

    rank = 1;
  }
  Bnds(Bnd b0, Bnd b1) {
    l[0] = b0.l;
    l[1] = b1.l;

    u[0] = b0.u;
    u[1] = b1.u;

    rank = 2;
  }
  Bnds(Bnd b0, Bnd b1, Bnd b2) {
    l[0] = b0.l;
    l[1] = b1.l;
    l[2] = b2.l;

    u[0] = b0.u;
    u[1] = b1.u;
    u[2] = b2.u;

    rank = 3;
  }
  Bnds(Bnd b0, Bnd b1, Bnd b2, Bnd b3) {
    l[0] = b0.l;
    l[1] = b1.l;
    l[2] = b2.l;
    l[3] = b3.l;

    u[0] = b0.u;
    u[1] = b1.u;
    u[2] = b2.u;
    u[3] = b3.u;

    rank = 4;
  }
  Bnds(Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4) {
    l[0] = b0.l;
    l[1] = b1.l;
    l[2] = b2.l;
    l[3] = b3.l;
    l[4] = b4.l;

    u[0] = b0.u;
    u[1] = b1.u;
    u[2] = b2.u;
    u[3] = b3.u;
    u[4] = b4.u;

    rank = 5;
  }
  Bnds(Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5) {
    l[0] = b0.l;
    l[1] = b1.l;
    l[2] = b2.l;
    l[3] = b3.l;
    l[4] = b4.l;
    l[5] = b5.l;

    u[0] = b0.u;
    u[1] = b1.u;
    u[2] = b2.u;
    u[3] = b3.u;
    u[4] = b4.u;
    u[5] = b5.u;

    rank = 6;
  }
  Bnds(Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5, Bnd b6) {
    l[0] = b0.l;
    l[1] = b1.l;
    l[2] = b2.l;
    l[3] = b3.l;
    l[4] = b4.l;
    l[5] = b5.l;
    l[6] = b6.l;

    u[0] = b0.u;
    u[1] = b1.u;
    u[2] = b2.u;
    u[3] = b3.u;
    u[4] = b4.u;
    u[5] = b5.u;
    u[6] = b6.u;

    rank = 7;
  }
  Bnds(Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5, Bnd b6, Bnd b7) {
    l[0] = b0.l;
    l[1] = b1.l;
    l[2] = b2.l;
    l[3] = b3.l;
    l[4] = b4.l;
    l[5] = b5.l;
    l[6] = b6.l;
    l[7] = b7.l;

    u[0] = b0.u;
    u[1] = b1.u;
    u[2] = b2.u;
    u[3] = b3.u;
    u[4] = b4.u;
    u[5] = b5.u;
    u[6] = b6.u;
    u[7] = b7.u;

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



#include "YAKL_CSArray.h"
template <class T, int rank, unsigned D0, unsigned D1=1, unsigned D2=1, unsigned D3=1>
using SArray  = Array< CSPEC< T , D0 , D1 , D2 , D3 > , rank , memStack , styleC >;

#include "YAKL_FSArray.h"
template <class T, int rank, class B0 , class B1=SB<1,1> , class B2=SB<1,1> , class B3=SB<1,1> >
using FSArray = Array< FSPEC< T , B0 , B1 , B2 , B3 > , rank , memStack , styleFortran >;
        
#include "YAKL_CArray.h"

#include "YAKL_FArray.h"



///////////////////////////////////////////////////////////
// Matrix multiplication routines for column-row format
///////////////////////////////////////////////////////////
template <class T, index_t COL_L, index_t ROW_L, index_t COL_R>
YAKL_INLINE SArray<T,2,COL_R,ROW_L>
matmul_cr ( SArray<T,2,COL_L,ROW_L> const &left ,
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
matmul_cr ( SArray<T,2,COL_L,ROW_L> const &left ,
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


template <class T, index_t COL_L, index_t ROW_L, index_t COL_R>
YAKL_INLINE FSArray<T,2,SB<COL_R>,SB<ROW_L>>
matmul_cr ( FSArray<T,2,SB<COL_L>,SB<ROW_L>> const &left ,
            FSArray<T,2,SB<COL_R>,SB<COL_L>> const &right ) {
  FSArray<T,2,SB<COL_R>,SB<ROW_L>> ret;
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
YAKL_INLINE FSArray<T,1,SB<ROW_L>>
matmul_cr ( FSArray<T,2,SB<COL_L>,SB<ROW_L>> const &left ,
            FSArray<T,1,SB<COL_L>>           const &right ) {
  FSArray<T,1,SB<ROW_L>> ret;
  for (index_t j=0; j < ROW_L; j++) {
    T tmp = 0;
    for (index_t k=0; k < COL_L; k++) {
      tmp += left(k,j) * right(k);
    }
    ret(j) = tmp;
  }
  return ret;
}


///////////////////////////////////////////////////////////
// Matrix multiplication routines for row-column format
///////////////////////////////////////////////////////////
template <class T, index_t COL_L, index_t ROW_L, index_t COL_R>
YAKL_INLINE SArray<T,2,ROW_L,COL_R>
matmul_rc ( SArray<T,2,ROW_L,COL_L> const &left ,
            SArray<T,2,COL_L,COL_R> const &right ) {
  SArray<T,2,ROW_L,COL_R> ret;
  for (index_t i=0; i < COL_R; i++) {
    for (index_t j=0; j < ROW_L; j++) {
      T tmp = 0;
      for (index_t k=0; k < COL_L; k++) {
        tmp += left(j,k) * right(k,i);
      }
      ret(j,i) = tmp;
    }
  }
  return ret;
}


template<class T, index_t COL_L, index_t ROW_L>
YAKL_INLINE SArray<T,1,ROW_L>
matmul_rc ( SArray<T,2,ROW_L,COL_L> const &left ,
            SArray<T,1,COL_L>       const &right ) {
  SArray<T,1,ROW_L> ret;
  for (index_t j=0; j < ROW_L; j++) {
    T tmp = 0;
    for (index_t k=0; k < COL_L; k++) {
      tmp += left(j,k) * right(k);
    }
    ret(j) = tmp;
  }
  return ret;
}


template <class T, index_t COL_L, index_t ROW_L, index_t COL_R>
YAKL_INLINE FSArray<T,2,SB<ROW_L>,SB<COL_R>>
matmul_rc ( FSArray<T,2,SB<ROW_L>,SB<COL_L>> const &left ,
            FSArray<T,2,SB<COL_L>,SB<COL_R>> const &right ) {
  FSArray<T,2,SB<ROW_L>,SB<COL_R>> ret;
  for (index_t i=0; i < COL_R; i++) {
    for (index_t j=0; j < ROW_L; j++) {
      T tmp = 0;
      for (index_t k=0; k < COL_L; k++) {
        tmp += left(j,k) * right(k,i);
      }
      ret(j,i) = tmp;
    }
  }
  return ret;
}


template<class T, index_t COL_L, index_t ROW_L>
YAKL_INLINE FSArray<T,1,SB<ROW_L>>
matmul_rc ( FSArray<T,2,SB<ROW_L>,SB<COL_L>> const &left ,
            FSArray<T,1,SB<COL_L>>           const &right ) {
  FSArray<T,1,SB<ROW_L>> ret;
  for (index_t j=0; j < ROW_L; j++) {
    T tmp = 0;
    for (index_t k=0; k < COL_L; k++) {
      tmp += left(j,k) * right(k);
    }
    ret(j) = tmp;
  }
  return ret;
}




/////////////////////////////////////////////////////////////////
// Matrix multiplication with Gaussian Elimination (no pivoting)
// for column-row format
/////////////////////////////////////////////////////////////////
template <unsigned int n, class real>
YAKL_INLINE SArray<real,2,n,n> matinv_ge_cr(SArray<real,2,n,n> &a) {
  SArray<real,2,n,n> inv;

  // Initialize inverse as identity
  for (int icol = 0; icol < n; icol++) {
    for (int irow = 0; irow < n; irow++) {
      if (icol == irow) {
        inv(icol,irow) = 1;
      } else {
        inv(icol,irow) = 0;
      }
    }
  }

  // Gaussian elimination to zero out lower
  for (int idiag = 0; idiag < n; idiag++) {
    // Divide out the diagonal component from the first row
    real factor = static_cast<real>(1)/a(idiag,idiag);
    for (int icol = idiag; icol < n; icol++) {
      a(icol,idiag) *= factor;
    }
    for (int icol = 0; icol < n; icol++) {
      inv(icol,idiag) *= factor;
    }
    for (int irow = idiag+1; irow < n; irow++) {
      real factor = a(idiag,irow);
      for (int icol = idiag; icol < n; icol++) {
        a  (icol,irow) -= factor * a  (icol,idiag);
      }
      for (int icol = 0; icol < n; icol++) {
        inv(icol,irow) -= factor * inv(icol,idiag);
      }
    }
  }

  // Gaussian elimination to zero out upper
  for (int idiag = n-1; idiag >= 1; idiag--) {
    for (int irow = 0; irow < idiag; irow++) {
      real factor = a(idiag,irow);
      for (int icol = irow+1; icol < n; icol++) {
        a  (icol,irow) -= factor * a  (icol,idiag);
      }
      for (int icol = 0; icol < n; icol++) {
        inv(icol,irow) -= factor * inv(icol,idiag);
      }
    }
  }

  return inv;
}


/////////////////////////////////////////////////////////////////
// Matrix multiplication with Gaussian Elimination (no pivoting)
// for row-column format
/////////////////////////////////////////////////////////////////
template <unsigned int n, class real>
YAKL_INLINE SArray<real,2,n,n> matinv_ge_rc(SArray<real,2,n,n> &a) {
  SArray<real,2,n,n> inv;

  // Initialize inverse as identity
  for (int icol = 0; icol < n; icol++) {
    for (int irow = 0; irow < n; irow++) {
      if (icol == irow) {
        inv(irow,icol) = 1;
      } else {
        inv(irow,icol) = 0;
      }
    }
  }

  // Gaussian elimination to zero out lower
  for (int idiag = 0; idiag < n; idiag++) {
    // Divide out the diagonal component from the first row
    real factor = static_cast<real>(1)/a(idiag,idiag);
    for (int icol = idiag; icol < n; icol++) {
      a(idiag,icol) *= factor;
    }
    for (int icol = 0; icol < n; icol++) {
      inv(idiag,icol) *= factor;
    }
    for (int irow = idiag+1; irow < n; irow++) {
      real factor = a(irow,idiag);
      for (int icol = idiag; icol < n; icol++) {
        a  (irow,icol) -= factor * a  (idiag,icol);
      }
      for (int icol = 0; icol < n; icol++) {
        inv(irow,icol) -= factor * inv(idiag,icol);
      }
    }
  }

  // Gaussian elimination to zero out upper
  for (int idiag = n-1; idiag >= 1; idiag--) {
    for (int irow = 0; irow < idiag; irow++) {
      real factor = a(irow,idiag);
      for (int icol = irow+1; icol < n; icol++) {
        a  (irow,icol) -= factor * a  (idiag,icol);
      }
      for (int icol = 0; icol < n; icol++) {
        inv(irow,icol) -= factor * inv(idiag,icol);
      }
    }
  }

  return inv;
}







