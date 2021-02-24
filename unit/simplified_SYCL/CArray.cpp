
#include <iostream>
#include "YAKL.h"

using yakl::Array;
using yakl::styleC;
using yakl::memHost;
using yakl::memDevice;
using yakl::c::parallel_for;
using yakl::c::SimpleBounds;

typedef float real;

typedef Array<real,1,memDevice,styleC> real1d;


void die(std::string msg) {
  std::cerr << msg << std::endl;
  exit(-1);
}


int main() {
  yakl::init();
  {
    int constexpr d1 = 2;

    ///////////////////////////////////////////////////////////
    // Test operator()
    ///////////////////////////////////////////////////////////

    real1d test1d("test1d",d1);

    parallel_for( SimpleBounds<1>(d1) , YAKL_LAMBDA (int i1) {
      test1d(i1) = 1;
    });

  }
  yakl::finalize();
  
  return 0;
}

