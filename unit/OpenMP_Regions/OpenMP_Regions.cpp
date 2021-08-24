
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

typedef Array<size_t,1,memHost,styleC> intHost1d;

typedef Array<size_t,1,memDevice,styleC> int1d;


void die(std::string msg) {
  std::cerr << msg << std::endl;
  exit(-1);
}


int main() {
  yakl::init();
  {
    int constexpr n1 = 100;
    int constexpr n2 = 1024;
    int1d sum_a("sum_a",n1);
    int1d sum_b("sum_b",n1);
    int1d sum_c("sum_c",n1);
    #pragma omp parallel for
    for (int i1=0; i1 < n1; i1++) {
      int1d a("a",n2);
      int1d b("b",n2);
      int1d c("c",n2);
      for (int i2=0; i2 < n2; i2++) {
        a(i2) = i1 + 1;
        b(i2) = i1 + 2;
        c(i2) = i1 + i2;
      }
      sum_a(i1) = yakl::intrinsics::minval(a);
      sum_b(i1) = yakl::intrinsics::minval(b);
      sum_c(i1) = yakl::intrinsics::minval(c);
    }
    for (int i1=0; i1 < n1; i1++) {
      std::cout << i1 << sum_a(i1) << " , " << sum_b(i1) << " , " << sum_c(i1) << "\n";
    }
  }
  yakl::finalize();
  
  return 0;
}

