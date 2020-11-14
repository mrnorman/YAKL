
#include "YAKL.h"
using yakl::SArray;
typedef double real;


template <unsigned int n>
void matrix_inverse(SArray<real,2,n,n> &a);


int main() {
  unsigned int constexpr n = 3;
  SArray<real,2,n,n> a;
  SArray<real,2,n,n> ainv;

  a(0,0) = 9;
  a(1,0) = 2;
  a(2,0) = 3;
  a(0,1) = 4;
  a(1,1) = 5;
  a(2,1) = 6;
  a(0,2) = 7;
  a(1,2) = 8;
  a(2,2) = 9;

  for (int i=0; i < n; i++) {
    for (int j=0; j < n; j++) {
      std::cout << std::setw(16) << a(j,i) << "  ";
    }
    std::cout << std::endl;
  }
  std::cout << "\n\n";

  matrix_inverse(a);

  for (int i=0; i < n; i++) {
    for (int j=0; j < n; j++) {
      std::cout << std::setw(16) << a(j,i) << "  ";
    }
    std::cout << std::endl;
  }
  std::cout << "\n\n";

  matrix_inverse(a);

  for (int i=0; i < n; i++) {
    for (int j=0; j < n; j++) {
      std::cout << std::setw(16) << a(j,i) << "  ";
    }
    std::cout << std::endl;
  }

}


template <unsigned int n>
void pentadiagonal(SArray<real,1,n> const &a,
                   SArray<real,1,n> const &b,
                   SArray<real,1,n> const &c,
                   SArray<real,1,n> const &d,
                   SArray<real,1,n> const &e,
                   SArray<real,1,n> const &f,
                   SArray<real,1,n> const &u) {
// solves for a vector u of length n in the pentadiagonal linear system
//  a_i u_(i-2) + b_i u_(i-1) + c_i u_i + d_i u_(i+1) + e_i u_(i+2) = f_i
// input are the a, b, c, d, e, and f and they are not modified

// in its clearest incarnation, this algorithm uses three storage arrays
// called p, q and r. here, the solution vector u is used for r, cutting
// the extra storage down to two arrays.
  SArray<real,1,n> p, q;
  real bet, den;

  // initialize elimination and backsubstitution arrays
  if (c(1) .eq. 0.0)  stop 'eliminate u2 trivially'
  bet  = 1.0d0/c(1)
  p(1) = -d(1) * bet
  q(1) = -e(1) * bet
  u(1) =  f(1) * bet

  bet = c(2) + b(2)*p(1)
  if (bet .eq. 0.0) stop 'bet singular in pentadiagonal'
  bet  = -1.0d0/bet
  p(2) = (d(2) + b(2)*q(1)) * bet
  q(2) = e(2) * bet
  u(2) = (b(2)*u(1) - f(2)) * bet

  // reduce to upper triangular
  do i=3,n
   bet = b(i) + a(i) * p(i-2)
   den = c(i) + a(i)*q(i-2) + bet*p(i-1)
   if (den .eq. 0.0) stop 'den singular in pentadiagonal'
   den = -1.0d0/den
   p(i) = (d(i) + bet*q(i-1)) * den
   q(i) = e(i) * den
   u(i) = (a(i)*u(i-2) + bet*u(i-1) - f(i)) * den
  enddo

  // backsubstitution
  u(n-1) = u(n-1) + p(n-1) * u(n)
  do i=n-2,1,-1
   u(i) = u(i) + p(i) * u(i+1) + q(i) * u(i+2)
  enddo
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


