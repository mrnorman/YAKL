
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

typedef Array<real,1,memHost,styleC> real1d;

void die(std::string msg) {
  std::cerr << msg << std::endl;
  exit(-1);
}


int main() {
  yakl::init();
  {
    int constexpr n = 1024*1024;
    real1d arr("arr",n);
    for (int i=0; i < n; i++) {
      yakl::Random rand(i);
      arr(i) = rand.genFP<real>();
    }
    real avg = yakl::intrinsics::sum(arr) / n;

    real var = 0;
    for (int i=0; i < n; i++) {
      real absdiff = abs(arr(i) - avg);
      var += absdiff * absdiff;
    }
    var = var / n;
    if (abs(avg-0.5)/0.5 > 0.01) { die("ERROR: mean is wrong"); }
    if (abs(var-(1./12.))/(1./12.) > 0.01) { die("ERROR: variance is wrong"); }
  }
  yakl::finalize();
  
  return 0;
}

