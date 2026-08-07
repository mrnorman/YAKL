
#include <iostream>
#include <utility>
#include "YAKL.h"

using yakl::Array;
using yakl::parallel_for;
using yakl::Bounds;
using yakl::SimpleBounds;
using yakl::COLON;


void die(std::string msg) {
  Kokkos::abort(msg.c_str());
}


template <class T>
void test_device_atomics(int n) {
  Array<T *,yakl::DeviceSpace> data("data",n);
  parallel_for( n , KOKKOS_LAMBDA (int i) {
    data(i) = i - (n-1)/2.;
  });

  yakl::ScalarLiveOut<T> min(99999);
  yakl::ScalarLiveOut<T> sum(0);
  yakl::ScalarLiveOut<T> max(-99999);
  parallel_for( n , KOKKOS_LAMBDA (int i) {
    Kokkos::atomic_min(&min(),data(i));
    Kokkos::atomic_add(&sum(),data(i));
    Kokkos::atomic_max(&max(),data(i));
  });

  if (abs(sum.hostRead()) > 1.e-13) die("ERROR: Wrong device sum");
  if (abs(min.hostRead()+(n-1)/2.) > 1.e-13) die("ERROR: Wrong device min");
  if (abs(max.hostRead()-(n-1)/2.) > 1.e-13) die("ERROR: Wrong device max");
}


int main() {
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");
    int constexpr n = 1024 + 1;
    test_device_atomics<float >(n);
    test_device_atomics<double>(n);
    test_device_atomics<int   >(n);

    {
      typedef int T;

      Array<T *,Kokkos::HostSpace> data("data",n);
      for (int i=0; i < n; i++) {
        data(i) = i - (n-1)/2.;
      }

      int min = 99999;
      int sum = 0;
      int max = -99999;
      for (int i=0; i < n; i++) {
        Kokkos::atomic_min(&min,data(i));
        Kokkos::atomic_add(&sum,data(i));
        Kokkos::atomic_max(&max,data(i));
      }
      
      if ( abs(sum) > 1.e-13 ) { die("ERROR: Wrong device sum"); }
      if ( abs(min + (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device min"); }
      if ( abs(max - (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device max"); }
    }

    // Exercise ScalarLiveOut independently of atomics, including its shared
    // copy semantics and device-side arithmetic assignment/get accessors.
    {
      yakl::ScalarLiveOut<int> live;
      live.hostWrite(7);
      if (live.hostRead() != 7) { die("ERROR: ScalarLiveOut default construction or hostWrite failed"); }

      yakl::ScalarLiveOut<int> copied(live);
      copied.hostWrite(-3);
      if (live.hostRead() != -3) { die("ERROR: ScalarLiveOut copy constructor did not share storage"); }

      yakl::ScalarLiveOut<int> copyAssigned(0);
      copyAssigned = copied;
      copyAssigned.hostWrite(11);
      if (live.hostRead() != 11) { die("ERROR: ScalarLiveOut copy assignment did not share storage"); }

      yakl::ScalarLiveOut<int> moved(std::move(copyAssigned));
      yakl::ScalarLiveOut<int> moveAssigned(0);
      moveAssigned = std::move(moved);
      parallel_for( "ScalarLiveOut accessors" , 1 , KOKKOS_LAMBDA (int) {
        moveAssigned = moveAssigned.get() + 5;
      });
      if (moveAssigned.hostRead() != 16) { die("ERROR: ScalarLiveOut move or device accessors failed"); }
    }

    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize(); 
  return 0;
}
