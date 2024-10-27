
#include <iostream>
#include "YAKL.h"

/////////////////////////////////////////////////////////////////////
// Most of YAKL_parallel_for_c.h is tested in the CArray unit test.
// This is to cover what was not covered in CArray
/////////////////////////////////////////////////////////////////////

using yakl::Array;
using yakl::styleFortran;
using yakl::memHost;
using yakl::memDevice;
using yakl::fortran::parallel_for;
using yakl::fortran::Bounds;
using yakl::fortran::SimpleBounds;
using yakl::COLON;
using yakl::intrinsics::sum;

typedef double real;

typedef Array<real,1,memDevice,styleFortran> real1d;
typedef Array<real,3,memDevice,styleFortran> real3d;


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
    parallel_for( "mylabel" , n1 , KOKKOS_LAMBDA (int i) {
      arr1d(i) = 1;
    });
    if ( abs(sum(arr1d) - n1) / (n1) > 1.e-13) die("ERROR: Wrong sum for arr1d");
    parallel_for( "mylabel" , SimpleBounds<3>(n3,n2,n1) , KOKKOS_LAMBDA (int k, int j, int i) {
      arr3d(i,j,k) = 1;
    });
    if ( abs(sum(arr3d) - (double) n1*n2*n3) / (double) (n1*n2*n3) > 1.e-13) die("ERROR: Wrong sum for arr3d");

    arr3d = 0.;
  }
  yakl::finalize();
  Kokkos::finalize(); 
  
  return 0;
}

