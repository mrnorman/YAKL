
#include "YAKL_pentadiagonal.h"
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
  SArray<double,1,n> d;
  SArray<double,1,n> e;
  SArray<double,1,n> x;
  SArray<double,1,n> y;

  // Set diagonal components
  a(0) = 0.0; a(1) = 0.0; a(2) = 0.2; a(3) = 0.9; a(4) = 4.0; a(5) = 2.2; a(6) = 1.1;
  b(0) = 0.0; b(1) = 3.6; b(2) = 2.8; b(3) = 3.1; b(4) = 6.7; b(5) = 1.2; b(6) = 0.1;
  c(0) = 4.1; c(1) = 2.2; c(2) = 6.2; c(3) = 8.5; c(4) = 3.8; c(5) = 3.7; c(6) = 2.1;
  d(0) = 0.4; d(1) = 1.0; d(2) = 5.0; d(3) = 4.9; d(4) = 2.3; d(5) = 5.1; d(6) = 0.0;
  e(0) = 0.5; e(1) = 6.1; e(2) = 2.9; e(3) = 4.5; e(4) = 0.7; e(5) = 0.0; e(6) = 0.0;

  // Set RHS
  y(0) = 6.4;
  y(1) = 35.4;
  y(2) = 58.9;
  y(3) = 96.6;
  y(4) = 76.5;
  y(5) = 72.7;
  y(6) = 20.8;

  yakl::pentadiagonal(a,b,c,d,e,y,x);

  for (int i=0; i < n; i++) {
    if (abs(x(i) - (i+1)) > 1.e-13) {
      die("pentadiagonal: wrong answer");
    }
  }

  // Add cyclic values
  a(0) = 0.2;
  b(0) = 0.5;
  a(1) = 0.4;
  e(5) = 0.1;
  d(6) = 0.9;
  e(6) = 1.0;

  // Set RHS
  y(0) = 11.1;
  y(1) = 38.2;
  y(2) = 58.9;
  y(3) = 96.6;
  y(4) = 76.5;
  y(5) = 72.8;
  y(6) = 23.7;

  yakl::pentadiagonal_periodic(a,b,c,d,e,y,x);

  for (int i=0; i < n; i++) {
    if (abs(x(i) - (i+1)) > 1.e-13) {
      die("pentadiagonal_periodic: wrong answer");
    }
  }
}


