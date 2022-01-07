
#include <iostream>
#include "YAKL.h"
#include "YAKL_fft.h"

using yakl::Array;
using yakl::styleC;
using yakl::memHost;
using yakl::memDevice;
using yakl::c::parallel_for;
using yakl::c::Bounds;
using yakl::c::SimpleBounds;
using yakl::COLON;

typedef double real;

typedef Array<real,1,memDevice,styleC> real1d;
typedef Array<real,1,memHost,styleC> realHost1d;


void die(std::string msg) {
  std::cerr << msg << std::endl;
  exit(-1);
}


int main() {
  yakl::init();
  {
    int constexpr n = 8;

    real1d data("data",n+2);
    parallel_for( n , YAKL_LAMBDA (int i) {
      data(i) = i+1;
    });

    realHost1d dataInit("dataInit",n+2);
    data.deep_copy_to(dataInit);
    yakl::fence();

    yakl::RealFFT1D<n,real> fft;
    fft.init(fft.trig);

    parallel_for( 1 , YAKL_LAMBDA (int i) {
      fft.forward(data,fft.trig);
    });

    realHost1d fftExact("fftExact",n+2);
    fftExact(0) =  3.6000000000000000e+01;
    fftExact(1) =  0.0000000000000000e+00;
    fftExact(2) = -4.0000000000000000e+00;
    fftExact(3) =  9.6568542494923797e+00;
    fftExact(4) = -4.0000000000000000e+00;
    fftExact(5) =  4.0000000000000000e+00;
    fftExact(6) = -4.0000000000000000e+00;
    fftExact(7) =  1.6568542494923799e+00;
    fftExact(8) = -4.0000000000000000e+00;
    fftExact(9) =  0.0000000000000000e+00;

    auto dataHost = data.createHostCopy();
    for (int i=0; i < n+2; i++) {
      if ( abs(dataHost(i) - fftExact(i)) > 1.e-13 ) { die("ERROR: forward gives wrong answer"); }
    }

    parallel_for( 1 , YAKL_LAMBDA (int i) {
      fft.inverse(data,fft.trig);
    });

    data.deep_copy_to(dataHost);
    yakl::fence();
    for (int i=0; i < n; i++) {
      if ( abs(dataHost(i) - dataInit(i)) > 1.e-13 ) { die("ERROR: backward gives wrong answer"); }
    }

  }
  yakl::finalize();
  
  return 0;
}

