
#include <iostream>
#include "YAKL.h"

using yakl::Array;
using yakl::styleC;
using yakl::memHost;
using yakl::memDevice;
using yakl::c::parallel_for;
using yakl::c::Bounds;
using yakl::c::SimpleBounds;
using yakl::COLON;

typedef float real;

typedef Array<real,1,memDevice,styleC> real1d;

real1d a, b, c;


void die(std::string msg) {
  std::cerr << msg << std::endl;
  exit(-1);
}


int main() {
  yakl::init();
  {
    int constexpr n = 100;
    auto &a = ::a;
    auto &b = ::b;
    auto &c = ::c;

    a = real1d("a",n);
    b = real1d("b",n);
    c = real1d("c",n);

    memset(a,0.f);
    memset(b,2.f);
    memset(c,3.f);

    parallel_for( Bounds<1>(n) , YAKL_LAMBDA (int i) {
      a(i) = b(i) + c(i);
    });

    if (abs(yakl::intrinsics::sum(a)/n - 5) < 1.e-6) {
      die("ERROR: sum is incorrect");
    }
  }
  yakl::finalize();
  
  return 0;
}

