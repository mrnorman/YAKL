
#include "YAKL.h"
using yakl::SArray;
typedef double real;


template <unsigned int n>
void matrix_inverse(SArray<real,2,n,n> &a);


template <unsigned int n>
void pentadiagonal(SArray<real,1,n> const &a,
                   SArray<real,1,n> const &b,
                   SArray<real,1,n> const &c,
                   SArray<real,1,n> const &d,
                   SArray<real,1,n> const &e,
                   SArray<real,1,n> const &f,
                   SArray<real,1,n> &u);


int main() {
  unsigned int constexpr n = 7;
  SArray<real,1,n> a;
  SArray<real,1,n> b;
  SArray<real,1,n> c;
  SArray<real,1,n> d;
  SArray<real,1,n> e;
  SArray<real,1,n> x;
  SArray<real,1,n> y;

  a(0) = 0.0; a(1) = 0.0; a(2) = 0.2; a(3) = 0.9; a(4) = 4.0; a(5) = 2.2; a(6) = 1.1;
  b(0) = 0.0; b(1) = 3.6; b(2) = 2.8; b(3) = 3.1; b(4) = 6.7; b(5) = 1.2; b(6) = 0.1;
  c(0) = 4.1; c(1) = 2.2; c(2) = 6.2; c(3) = 8.5; c(4) = 3.8; c(5) = 3.7; c(6) = 2.1;
  d(0) = 0.4; d(1) = 1.0; d(2) = 5.0; d(3) = 4.9; d(4) = 2.3; d(5) = 5.1; d(6) = 0.0;
  e(0) = 0.5; e(1) = 6.1; e(2) = 2.9; e(3) = 4.5; e(4) = 0.7; e(5) = 0.0; e(6) = 0.0;

  y(0) = 6.4;
  y(1) = 35.4;
  y(2) = 58.9;
  y(3) = 96.6;
  y(4) = 76.5;
  y(5) = 72.7;
  y(6) = 20.8;

  pentadiagonal(a,b,c,d,e,y,x);

  for (int i=0; i < n; i++) {
    std::cout << x(i) << "\n";
  }

}


template <unsigned int n>
void pentadiagonal(SArray<real,1,n> const &a,
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



template <unsigned int n>
void matrix_inverse(SArray<real,2,n,n> &a) {
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


