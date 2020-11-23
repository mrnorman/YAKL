
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

namespace blah {
  real1d a, b, c;
}


void die(std::string msg) {
  std::cerr << msg << std::endl;
  exit(-1);
}


int main() {
  yakl::init();
  {
    int constexpr n = 100;
    auto a = blah::a;
    auto b = blah::b;
    auto c = blah::c;

    a = real1d("a",n);
    b = real1d("b",n);
    c = real1d("c",n);

    memset(a,0.f);
    memset(b,2.f);
    memset(c,3.f);

    parallel_for( Bounds<1>(n) , YAKL_LAMBDA (int i) {
      a(i) = b(i) + c(i);
    });

    blah::a = a;
    blah::b = b;
    blah::c = c;

    if (abs(yakl::intrinsics::sum(blah::a)/n - 5) > 1.e-6) {
      die("ERROR: sum is incorrect");
    }

    blah::a = real1d();
    blah::b = real1d();
    blah::c = real1d();
  }
  yakl::finalize();
  
  return 0;
}

