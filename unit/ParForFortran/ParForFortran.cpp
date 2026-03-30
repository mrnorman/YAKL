
#include <iostream>
#include "YAKL.h"

/////////////////////////////////////////////////////////////////////
// Most of YAKL_parallel_for_c.h is tested in the CArray unit test.
// This is to cover what was not covered in CArray
/////////////////////////////////////////////////////////////////////

using yakl::Array_F;
using yakl::parallel_for_F;
using yakl::Bounds_F;
using yakl::SimpleBounds_F;
using yakl::COLON;
using yakl::intrinsics::sum;

typedef double real;

typedef Array_F<real *  ,yakl::DeviceSpace> real1d;
typedef Array_F<real ***,yakl::DeviceSpace> real3d;


void die(std::string msg) {
  Kokkos::abort(msg.c_str());
}


int main() {
  Kokkos::initialize();
  yakl::init();
  {
    int constexpr n1 = 4;
    int constexpr n2 = 16;
    int constexpr n3 = 128;
    real1d arr1d("arr1d",n1);
    real3d arr3d("arr3d",n1,n2,n3);
    // Test with labels and SimpleBounds
    parallel_for_F( "mylabel" , n1 , KOKKOS_LAMBDA (int i) {
      arr1d(i) = 1;
    });
    if ( abs(sum(arr1d) - n1) / (n1) > 1.e-13) die("ERROR: Wrong sum for arr1d");
    parallel_for_F( "mylabel" , SimpleBounds_F<3>(n3,n2,n1) , KOKKOS_LAMBDA (int k, int j, int i) {
      arr3d(i,j,k) = 1;
    });
    if ( abs(sum(arr3d) - (double) n1*n2*n3) / (double) (n1*n2*n3) > 1.e-13) die("ERROR: Wrong sum for arr3d");

    arr3d = 0.;
  }
  yakl::finalize();
  Kokkos::finalize(); 
  
  return 0;
}

