
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

    blah::a = real1d("a",n);
    blah::b = real1d("b",n);
    blah::c = real1d("c",n);

    memset(blah::a,0.f);
    memset(blah::b,2.f);
    memset(blah::c,3.f);

    YAKL_SCOPE(a,::blah::a);
    YAKL_SCOPE(b,::blah::b);
    YAKL_SCOPE(c,::blah::c);

    parallel_for( Bounds<1>(n) , YAKL_LAMBDA (int i) {
      a(i) = b(i) + c(i);
    });

    if (abs(yakl::intrinsics::sum(blah::a)/n - 5) > 1.e-6) {
      std::cout << yakl::intrinsics::sum(blah::a)/n << "\n";
      die("ERROR: sum is incorrect");
    }

    blah::a = real1d();
    blah::b = real1d();
    blah::c = real1d();
  }
  yakl::finalize();
  
  return 0;
}

