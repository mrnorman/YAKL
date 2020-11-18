
#include "YAKL.h"
using yakl::SArray;


namespace yakl {


  /*
  Solves a tridiagonal system with periodic boundary conditions of the form:

  [b(0)   c(0)  0    0      0     a(0)  ] [x(0)  ] = [d(0)  ]
  [a(1)   b(1) c(1)  0      0      0    ] [x(1)  ] = [d(1)  ]
  [ 0     a(2) b(2) c(2)    0      0    ] [x(2)  ] = [d(2)  ]
  [ 0      0   ..  ..  ..   0      0    ] [ .    ] = [ .    ]
  [ 0      0      ..  ..  ..       0    ] [ .    ] = [ .    ]
  [ 0      0    0   a(n-2) b(n-2) c(n-2)] [x(n-2)] = [d(n-2)]
  [c(n-1)  0    0    0     a(n-1) b(n-1)] [x(n-1)] = [d(n-1)]

  This routine stores the result in d(), and as the signature indicates, it overwrites b, c, d

  This uses the Thomas algorithm with the Sherman-Morrison formula.
  The Sherman-Morrison Formula is as follows:

  Separate the tridiagonal + periodic matrix, A, into (B + u*v^T),
  where B is strictly tridiagonal, and u*v^T accounts for the non-tridiagonal periodic BCs:

  u = [-b(0) , 0 , ... , 0 , c(n-1)    ]^T
  v = [1     , 0 , ... , 0 , -a(0)/b(0)]^T

  Now we're solveing the system (B + u*v^T)*x = d, which is identical to A*x=d.

  To get the solution, we solve two systems:

  (1) B*y=d
  (2) B*q=u

  In this code, q is labeled as "tmp". Then, the answer is given by:

  x = y - ( (v^T*y) / (1 + v^T*q) ) * q

  Unfortunately, periodic boundary conditions roughly double the amount of work in the tridiagonal solve

  */
  template <class real, unsigned int n>
  void tridiagonal_periodic(SArray<real,1,n> const &a, SArray<real,1,n> &b, SArray<real,1,n> &c, SArray<real,1,n> &d) {
    SArray<real,1,n> tmp;
    // Save the original "b0" because it's needed later on to compute ( (v^T*y) / (1 + v^T*q) )
    real b0 = b(0);

    // This is the vector "u"
    tmp(0  ) = -b0;
    tmp(n-1) =  c(n-1);

    // The new tridiagonal system "B" alters the entries of the main diagonal
    b(n-1) = b(n-1) + a(0)*c(n-1)/b(0);
    b(0  ) = b(0  ) + b(0);

    // Thomas algorithm for  B*y=d  and  B*q=u simultaneously to save cost
    real div = static_cast<real>(1) / b(0);
    c  (0) *= div;
    d  (0) *= div;
    tmp(0) *= div;
    for (int i = 1; i < n-1; i++) {
      div = static_cast<real>(1) / (b(i) - a(i)*c(i-1));
      c  (i) =  c(i)                  * div;
      d  (i) = (d(i) - a(i)*d  (i-1)) * div;
      tmp(i) = (     - a(i)*tmp(i-1)) * div;
    }
    div = static_cast<real>(1) / (b(n-1) - a(n-1)*c(n-2));
    d  (n-1) = (d  (n-1) - a(n-1)*d  (n-2)) * div;
    tmp(n-1) = (tmp(n-1) - a(n-1)*tmp(n-2)) * div;
    for (int i = n-2; i >= 0; i--) {
      d  (i) -= c(i)*d  (i+1);
      tmp(i) -= c(i)*tmp(i+1);
    }

    // Compute factor = ( (v^T*y) / (1 + v^T*q) )
    real factor;
    if ( (tmp(0) - a(0)*tmp(n-1)/b0 + static_cast<real>(1)) == 0 ) {
      factor = 0;
    } else {
      factor = (d(0) - a(0)*d(n-1)/b0)/(tmp(0) - a(0)*tmp(n-1)/b0 + static_cast<real>(1));
    }

    // Subtract factor * q from the previous solution to get the final solution
    for (int i = 0; i < n; i++) {
      d(i) -= factor*tmp(i);
    }
  }



  /*
  Solves a tridiagonal system with no boundary conditions of the form:

  [b(0)   c(0)  0    0      0      0    ] [x(0)  ] = [d(0)  ]
  [a(1)   b(1) c(1)  0      0      0    ] [x(1)  ] = [d(1)  ]
  [ 0     a(2) b(2) c(2)    0      0    ] [x(2)  ] = [d(2)  ]
  [ 0      0   ..  ..  ..   0      0    ] [ .    ] = [ .    ]
  [ 0      0      ..  ..  ..       0    ] [ .    ] = [ .    ]
  [ 0      0    0   a(n-2) b(n-2) c(n-2)] [x(n-2)] = [d(n-2)]
  [ 0      0    0    0     a(n-1) b(n-1)] [x(n-1)] = [d(n-1)]

  This routine stores the result in d(), and as the signature indicates, it overwrites b, c, d.

  This uses the Thomas algorithm.
  */
  template <class real, unsigned int n>
  void tridiagonal(SArray<real,1,n> const &a, SArray<real,1,n> const &b, SArray<real,1,n> &c, SArray<real,1,n> &d) {
    real tmp = static_cast<real>(1) / b(0);
    c(0) *= tmp;
    d(0) *= tmp;
    for (int i = 1; i < n-1; i++) {
      real tmp = static_cast<real>(1) / (b(i) - a(i)*c(i-1));
      c(i) *= tmp;
      d(i) = (d(i) - a(i)*d(i-1)) * tmp;
    }
    d(n-1) = (d(n-1) - a(n-1)*d(n-2)) / (b(n-1) - a(n-1)*c(n-2));
    for (int i = n-2; i >= 0; i--) {
      d(i) -= c(i)*d(i+1);
    }
  }

}


