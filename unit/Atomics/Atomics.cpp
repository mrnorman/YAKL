
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
  Kokkos::abort(msg.c_str());
}


int main() {
  Kokkos::initialize();
  yakl::init();
  {
    int constexpr n = 1024 + 1;
    {
      typedef float T;

      Array<T,1,memDevice,styleC> data("data",n);
      parallel_for( n , KOKKOS_LAMBDA (int i) {
        data(i) = i - (n-1)/2.;
      });

      yakl::ScalarLiveOut<T> min(99999);
      parallel_for( n , KOKKOS_LAMBDA (int i) {
        Kokkos::atomic_min(&min(),data(i));
      });

      yakl::ScalarLiveOut<T> sum(0.);
      parallel_for( n , KOKKOS_LAMBDA (int i) {
        Kokkos::atomic_add(&sum(),data(i));
      });

      yakl::ScalarLiveOut<T> max(-99999);
      parallel_for( n , KOKKOS_LAMBDA (int i) {
        Kokkos::atomic_max(&max(),data(i));
      });
      
      if ( abs(sum.hostRead()) > 1.e-13 ) { die("ERROR: Wrong device sum"); }
      if ( abs(min.hostRead() + (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device min"); }
      if ( abs(max.hostRead() - (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device max"); }
    }

    {
      typedef double T;

      Array<T,1,memDevice,styleC> data("data",n);
      parallel_for( n , KOKKOS_LAMBDA (int i) {
        data(i) = i - (n-1)/2.;
      });

      yakl::ScalarLiveOut<T> min(99999);
      parallel_for( n , KOKKOS_LAMBDA (int i) {
        Kokkos::atomic_min(&min(),data(i));
      });

      yakl::ScalarLiveOut<T> sum(0.);
      parallel_for( n , KOKKOS_LAMBDA (int i) {
        Kokkos::atomic_add(&sum(),data(i));
      });

      yakl::ScalarLiveOut<T> max(-99999);
      parallel_for( n , KOKKOS_LAMBDA (int i) {
        Kokkos::atomic_max(&max(),data(i));
      });
      
      if ( abs(sum.hostRead()) > 1.e-13 ) { die("ERROR: Wrong device sum"); }
      if ( abs(min.hostRead() + (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device min"); }
      if ( abs(max.hostRead() - (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device max"); }
    }

    {
      typedef int T;

      Array<T,1,memDevice,styleC> data("data",n);
      parallel_for( n , KOKKOS_LAMBDA (int i) {
        data(i) = i - (n-1)/2.;
      });

      yakl::ScalarLiveOut<T> min(99999);
      parallel_for( n , KOKKOS_LAMBDA (int i) {
        Kokkos::atomic_min(&min(),data(i));
      });

      yakl::ScalarLiveOut<T> sum(0.);
      parallel_for( n , KOKKOS_LAMBDA (int i) {
        Kokkos::atomic_add(&sum(),data(i));
      });

      yakl::ScalarLiveOut<T> max(-99999);
      parallel_for( n , KOKKOS_LAMBDA (int i) {
        Kokkos::atomic_max(&max(),data(i));
      });
      
      if ( abs(sum.hostRead()) > 1.e-13 ) { die("ERROR: Wrong device sum"); }
      if ( abs(min.hostRead() + (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device min"); }
      if ( abs(max.hostRead() - (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device max"); }
    }

    {
      typedef int T;

      Array<T,1,yakl::memHost,styleC> data("data",n);
      for (int i=0; i < n; i++) {
        data(i) = i - (n-1)/2.;
      }

      int min = 99999;
      for (int i=0; i < n; i++) {
        Kokkos::atomic_min(&min,data(i));
      }

      int sum = 0;
      for (int i=0; i < n; i++) {
        Kokkos::atomic_add(&sum,data(i));
      }

      int max = -99999;
      for (int i=0; i < n; i++) {
        Kokkos::atomic_max(&max,data(i));
      }
      
      if ( abs(sum) > 1.e-13 ) { die("ERROR: Wrong device sum"); }
      if ( abs(min + (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device min"); }
      if ( abs(max - (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device max"); }
    }

    {
      typedef double real;
      int constexpr n = 1024*16;
      Array<real,1,yakl::memDevice,styleC> data("data",n);
      parallel_for( n , KOKKOS_LAMBDA (int i) { data(i) = yakl::Random(i).genFP<real>(); });
      for (int k=0; k < 10; k++) {
        yakl::ScalarLiveOut<real> sum(0.);
        parallel_for( n , KOKKOS_LAMBDA (int i) { Kokkos::atomic_add( &sum() , data(i) ); });
        std::cout << std::scientific << std::setprecision(18) << sum.hostRead() << "\n";
      }
    }

  }
  yakl::finalize();
  Kokkos::finalize(); 
  return 0;
}

