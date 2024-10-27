
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

typedef double real;

typedef Array<real,1,memHost,styleC> realHost1d;

typedef Array<real,1,memDevice,styleC> real1d;


void die(std::string msg) {
  Kokkos::abort(msg.c_str());
}


int main() {
  Kokkos::initialize();
  yakl::init();
  {
    int constexpr n = 1024*1024 + 1;
    real1d data("data",n);
    parallel_for( "Initialize data" , n , KOKKOS_LAMBDA (int i) {
      data(i) = i - (n-1)/2.;
    });
    real sum = yakl::intrinsics::sum   ( data );
    real min = yakl::intrinsics::minval( data );
    real max = yakl::intrinsics::maxval( data );
    if ( abs(sum) > 1.e-13 ) { die("ERROR: Wrong device sum"); }
    if ( abs(min + (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device min"); }
    if ( abs(max - (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device max"); }

    auto dataHost = data.createHostCopy();
    sum = yakl::intrinsics::sum   ( dataHost );
    min = yakl::intrinsics::minval( dataHost );
    max = yakl::intrinsics::maxval( dataHost );
    if ( abs(sum) > 1.e-13 ) { die("ERROR: Wrong device sum"); }
    if ( abs(min + (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device min"); }
    if ( abs(max - (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device max"); }
  }
  yakl::finalize();
  Kokkos::finalize(); 
  
  return 0;
}

