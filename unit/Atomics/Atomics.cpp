
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
  std::cerr << msg << std::endl;
  exit(-1);
}


int main() {
  yakl::init();
  {
    int constexpr n = 1024*1024 + 1;
    real1d data("data",n);
    parallel_for( n , YAKL_LAMBDA (int i) {
      data(i) = i - (n-1)/2.;
    });

    yakl::ScalarLiveOut<real> min(std::numeric_limits<real>::max());
    parallel_for( n , YAKL_LAMBDA (int i) {
      yakl::atomicMin(min(),data(i));
    });

    yakl::ScalarLiveOut<real> sum(0.);
    parallel_for( n , YAKL_LAMBDA (int i) {
      yakl::atomicAdd(sum(),data(i));
    });

    yakl::ScalarLiveOut<real> max(std::numeric_limits<real>::lowest());
    parallel_for( n , YAKL_LAMBDA (int i) {
      yakl::atomicMax(max(),data(i));
    });
    
    if ( abs(sum.hostRead()) > 1.e-13 ) { die("ERROR: Wrong device sum"); }
    //if ( abs(min.hostRead() + (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device min"); }
    //if ( abs(max.hostRead() - (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device max"); }

  }
  yakl::finalize();
  
  return 0;
}

