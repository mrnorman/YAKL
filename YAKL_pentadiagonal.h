
#include "YAKL.h"
using yakl::SArray;


namespace yakl {



  template <unsigned int n, class real>
  YAKL_INLINE real penta_sum(SArray<real,1,n> const &v, SArray<real,1,n> const &z) {
    real sum = 0;
    for (int k=0; k < n; k++) {
      sum += v(k)*z(k);
    }
    return sum;
  }



  template <unsigned int n, class real>
  YAKL_INLINE void matrix_inverse_small(SArray<real,2,n,n> &a) {
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

    // Overwrite matrix with inverse
    for (int icol = 0; icol < n; icol++) {
      for (int irow = 0; irow < n; irow++) {
        a(icol,irow) = inv(icol,irow);
      }
    }
  }



  template <unsigned int n, class real>
  YAKL_INLINE void pentadiagonal(SArray<real,1,n> const &a,
                                 SArray<real,1,n> const &b,
                                 SArray<real,1,n> const &c,
                                 SArray<real,1,n> const &d,
                                 SArray<real,1,n> const &e,
                                 SArray<real,1,n> const &f,
                                 SArray<real,1,n> &u) {
    // solves for a vector u of length n in the pentadiagonal linear system
    //  a_i u_(i-2) + b_i u_(i-1) + c_i u_i + d_i u_(i+1) + e_i u_(i+2) = f_i
    // input are the a, b, c, d, e, and f and they are not modified

    // in its clearest incarnation, this algorithm uses three storage arrays
    // called p, q and r. here, the solution vector u is used for r, cutting
    // the extra storage down to two arrays.
    SArray<real,1,n> p, q;
    real bet, den;

    // initialize elimination and backsubstitution arrays
    bet  = static_cast<real>(1)/c(0);
    p(0) = -d(0) * bet;
    q(0) = -e(0) * bet;
    u(0) =  f(0) * bet;

    bet = c(1) + b(1)*p(0);
    bet  = -static_cast<real>(1)/bet;
    p(1) = (d(1) + b(1)*q(0)) * bet;
    q(1) = e(1) * bet;
    u(1) = (b(1)*u(0) - f(1)) * bet;

    // reduce to upper triangular
    for (int i=2; i < n; i++) {
      bet = b(i) + a(i) * p(i-2);
      den = c(i) + a(i)*q(i-2) + bet*p(i-1);
      den = -static_cast<real>(1)/den;
      p(i) = (d(i) + bet*q(i-1)) * den;
      q(i) = e(i) * den;
      u(i) = (a(i)*u(i-2) + bet*u(i-1) - f(i)) * den;
    }

    // backsubstitution
    u(n-2) = u(n-2) + p(n-2) * u(n-1);
    for (int i=n-3; i >= 0; i--) {
      u(i) = u(i) + p(i) * u(i+1) + q(i) * u(i+2);
    }
  }



  template <unsigned int n, class real>
  YAKL_INLINE void pentadiagonal_periodic(SArray<real,1,n> const &a,
                                          SArray<real,1,n> const &b,
                                          SArray<real,1,n> const &c,
                                          SArray<real,1,n> const &d,
                                          SArray<real,1,n> const &e,
                                          SArray<real,1,n> const &f,
                                          SArray<real,1,n> &x) {

    SArray<real,1,n>   u1, u2, u3, u4, v1, v2, v3, v4, z1, z2, z3, z4, r, s, y;
    SArray<real,2,4,4> h, p;

    real cp1 = a(0);
    real cp2 = b(0);
    real cp3 = a(1);
    real cp4 = e(n-2);
    real cp5 = d(n-1);
    real cp6 = e(n-1);
    
    for (int i=0; i < n; i++) {
      u1(i) = 0;   u2(i) = 0;   u3(i) = 0;   u4(i) = 0;
      v1(i) = 0;   v2(i) = 0;   v3(i) = 0;   v4(i) = 0;
      z1(i) = 0;   z2(i) = 0;   z3(i) = 0;   z4(i) = 0;
    }

    u1(0  ) = 1;
    u2(1  ) = 1;
    u3(n-2) = 1;
    u4(n-1) = 1;

    v1(n-2) = cp1;
    v1(n-1) = cp2;
    v2(n-1) = cp3;
    v3(0  ) = cp4;
    v4(0  ) = cp5;
    v4(1  ) = cp6;

    pentadiagonal(a,b,c,d,e,u1,z1);
    pentadiagonal(a,b,c,d,e,u2,z2);
    pentadiagonal(a,b,c,d,e,u3,z3);
    pentadiagonal(a,b,c,d,e,u4,z4);
    pentadiagonal(a,b,c,d,e,f ,y );

    p(0,0) = penta_sum(v1,z1);
    p(0,1) = penta_sum(v1,z2);
    p(0,2) = penta_sum(v1,z3);
    p(0,3) = penta_sum(v1,z4);

    p(1,0) = penta_sum(v2,z1);
    p(1,1) = penta_sum(v2,z2);
    p(1,2) = penta_sum(v2,z3);
    p(1,3) = penta_sum(v2,z4);

    p(2,0) = penta_sum(v3,z1);
    p(2,1) = penta_sum(v3,z2);
    p(2,2) = penta_sum(v3,z3);
    p(2,3) = penta_sum(v3,z4);

    p(3,0) = penta_sum(v4,z1);
    p(3,1) = penta_sum(v4,z2);
    p(3,2) = penta_sum(v4,z3);
    p(3,3) = penta_sum(v4,z4);

    for (int i=0; i < 4; i++) {
      p(i,i) = p(i,i) + 1;
    }

    matrix_inverse_small(p);

    r(0) = 0;
    r(1) = 0;
    r(2) = 0;
    r(3) = 0;
    for (int k=0; k < n; k++) {
      r(0) = r(0) + v1(k) * y(k);
      r(1) = r(1) + v2(k) * y(k);
      r(2) = r(2) + v3(k) * y(k);
      r(3) = r(3) + v4(k) * y(k);
    }

    for (int j=0; j < 4; j++) {
      s(j) = 0;
      for (int k=0; k < 4; k++) {
        s(j) = s(j) + p(j,k) * r(k);
      }
    }

    for (int j=0; j < n; j++) {
      real sum = z1(j)*s(0) + z2(j)*s(1) + z3(j)*s(2) + z4(j)*s(3);
      x(j) = y(j) - sum;
    }
  }

}


