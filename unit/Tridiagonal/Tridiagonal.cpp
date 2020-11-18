
#include "YAKL_tridiagonal.h"
using yakl::SArray;


void die(std::string msg) {
  std::cerr << msg << std::endl;
  exit(-1);
}


int main() {
  unsigned int constexpr n = 7;
  SArray<double,1,n> a;
  SArray<double,1,n> b;
  SArray<double,1,n> c;
  SArray<double,1,n> x;

  // Set diagonal components
  a(0) = 0.0; a(1) = 3.6; a(2) = 2.8; a(3) = 3.1; a(4) = 6.7; a(5) = 1.2; a(6) = 0.1;
  b(0) = 4.1; b(1) = 2.2; b(2) = 6.2; b(3) = 8.5; b(4) = 3.8; b(5) = 3.7; b(6) = 2.1;
  c(0) = 0.4; c(1) = 1.0; c(2) = 5.0; c(3) = 4.9; c(4) = 2.3; c(5) = 5.1; c(6) = 0.0;

  // Set RHS
  x(0) = 4.9;
  x(1) = 11;
  x(2) = 44.2;
  x(3) = 67.8;
  x(4) = 59.6;
  x(5) = 63.9;
  x(6) = 15.3;

  yakl::tridiagonal(a,b,c,x);

  for (int i=0; i < n; i++) {
    if (abs(x(i) - (i+1)) > 1.e-13) {
      die("tridiagonal: wrong answer");
    }
  }

  a(0) = 0.0; a(1) = 3.6; a(2) = 2.8; a(3) = 3.1; a(4) = 6.7; a(5) = 1.2; a(6) = 0.1;
  b(0) = 4.1; b(1) = 2.2; b(2) = 6.2; b(3) = 8.5; b(4) = 3.8; b(5) = 3.7; b(6) = 2.1;
  c(0) = 0.4; c(1) = 1.0; c(2) = 5.0; c(3) = 4.9; c(4) = 2.3; c(5) = 5.1; c(6) = 0.0;

  // Add cyclic values
  a(0) = 0.5;
  c(6) = 0.9;

  // Set RHS
  x(0) = 8.4;
  x(1) = 11;
  x(2) = 44.2;
  x(3) = 67.8;
  x(4) = 59.6;
  x(5) = 63.9;
  x(6) = 16.2;

  yakl::tridiagonal_periodic(a,b,c,x);

  for (int i=0; i < n; i++) {
    if (abs(x(i) - (i+1)) > 1.e-13) {
      die("tridiagonal_periodic: wrong answer");
    }
  }
}


