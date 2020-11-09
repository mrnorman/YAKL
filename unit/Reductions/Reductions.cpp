
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
    yakl::ParallelSum<real,memDevice> psum(data.totElems());
    real sum = psum( data.data() );

    yakl::ParallelMin<real,memDevice> pmin(data.totElems());
    real min = pmin( data.data() );

    yakl::ParallelMax<real,memDevice> pmax(data.totElems());
    real max = pmax( data.data() );
    
    if ( abs(sum) > 1.e-13 ) { die("ERROR: Wrong device sum"); }
    if ( abs(min + (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device min"); }
    if ( abs(max - (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device max"); }

    auto dataHost = data.createHostCopy();
    yakl::ParallelSum<real,memHost> psumHost(dataHost.totElems());
    sum = psumHost( dataHost.data() );

    yakl::ParallelMin<real,memHost> pminHost(dataHost.totElems());
    min = pminHost( dataHost.data() );

    yakl::ParallelMax<real,memHost> pmaxHost(dataHost.totElems());
    max = pmaxHost( dataHost.data() );
    
    if ( abs(sum) > 1.e-13 ) { die("ERROR: Wrong device sum"); }
    if ( abs(min + (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device min"); }
    if ( abs(max - (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device max"); }
  }
  yakl::finalize();
  
  return 0;
}

