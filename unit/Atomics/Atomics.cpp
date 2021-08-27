
#include <iostream>
#include "YAKL.h"

using yakl::Array;
using yakl::styleC;
using yakl::memDevice;
using yakl::c::parallel_for;
using yakl::c::Bounds;
using yakl::c::SimpleBounds;
using yakl::COLON;


void die(std::string msg) {
  std::cerr << msg << std::endl;
  exit(-1);
}


int main() {
  yakl::init();
  {
    int constexpr n = 1024 + 1;
    {
      typedef float T;

      Array<T,1,memDevice,styleC> data("data",n);
      parallel_for( n , YAKL_LAMBDA (int i) {
        data(i) = i - (n-1)/2.;
      });

      yakl::ScalarLiveOut<T> min(99999);
      parallel_for( n , YAKL_DEVICE_LAMBDA (int i) {
        yakl::atomicMin(min(),data(i));
      });

      yakl::ScalarLiveOut<T> sum(0.);
      parallel_for( n , YAKL_DEVICE_LAMBDA (int i) {
        yakl::atomicAdd(sum(),data(i));
      });

      yakl::ScalarLiveOut<T> max(-99999);
      parallel_for( n , YAKL_DEVICE_LAMBDA (int i) {
        yakl::atomicMax(max(),data(i));
      });
      
      if ( abs(sum.hostRead()) > 1.e-13 ) { die("ERROR: Wrong device sum"); }
      if ( abs(min.hostRead() + (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device min"); }
      if ( abs(max.hostRead() - (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device max"); }
    }

    {
      typedef double T;

      Array<T,1,memDevice,styleC> data("data",n);
      parallel_for( n , YAKL_LAMBDA (int i) {
        data(i) = i - (n-1)/2.;
      });

      yakl::ScalarLiveOut<T> min(99999);
      parallel_for( n , YAKL_DEVICE_LAMBDA (int i) {
        yakl::atomicMin(min(),data(i));
      });

      yakl::ScalarLiveOut<T> sum(0.);
      parallel_for( n , YAKL_DEVICE_LAMBDA (int i) {
        yakl::atomicAdd(sum(),data(i));
      });

      yakl::ScalarLiveOut<T> max(-99999);
      parallel_for( n , YAKL_DEVICE_LAMBDA (int i) {
        yakl::atomicMax(max(),data(i));
      });
      
      if ( abs(sum.hostRead()) > 1.e-13 ) { die("ERROR: Wrong device sum"); }
      if ( abs(min.hostRead() + (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device min"); }
      if ( abs(max.hostRead() - (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device max"); }
    }

    {
      typedef int T;

      Array<T,1,memDevice,styleC> data("data",n);
      parallel_for( n , YAKL_LAMBDA (int i) {
        data(i) = i - (n-1)/2.;
      });

      yakl::ScalarLiveOut<T> min(99999);
      parallel_for( n , YAKL_DEVICE_LAMBDA (int i) {
        yakl::atomicMin(min(),data(i));
      });

      yakl::ScalarLiveOut<T> sum(0.);
      parallel_for( n , YAKL_DEVICE_LAMBDA (int i) {
        yakl::atomicAdd(sum(),data(i));
      });

      yakl::ScalarLiveOut<T> max(-99999);
      parallel_for( n , YAKL_DEVICE_LAMBDA (int i) {
        yakl::atomicMax(max(),data(i));
      });
      
      if ( abs(sum.hostRead()) > 1.e-13 ) { die("ERROR: Wrong device sum"); }
      if ( abs(min.hostRead() + (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device min"); }
      if ( abs(max.hostRead() - (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device max"); }
    }

  }
  yakl::finalize();
  
  return 0;
}

