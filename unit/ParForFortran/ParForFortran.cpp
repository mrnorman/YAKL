
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
using yakl::c::parallel_outer;
using yakl::c::parallel_inner;
using yakl::c::single_inner;
using yakl::LaunchConfig;
using yakl::fortran::parallel_for;
using yakl::fortran::Bounds;
using yakl::fortran::SimpleBounds;
using yakl::COLON;
using yakl::fence_inner;
using yakl::intrinsics::sum;

typedef float real;

typedef Array<real,1,memDevice,styleFortran> real1d;
typedef Array<real,3,memDevice,styleFortran> real3d;


void die(std::string msg) {
  std::cerr << msg << std::endl;
  exit(-1);
}


int main() {
  yakl::init();
  {
    int constexpr n1 = 128;
    int constexpr n2 = 16;
    int constexpr n3 = 4;
    real1d arr1d("arr1d",n1);
    real3d arr3d("arr3d",n1,n2,n3);
    // Test with labels and SimpleBounds
    parallel_for( "mylabel" , n1 , YAKL_LAMBDA (int i) {
      arr1d(i) = 1;
    });
    if ( abs(sum(arr1d) - n1) / (n1) > 1.e-13) die("ERROR: Wrong sum for arr1d");
    parallel_for( "mylabel" , SimpleBounds<3>(n3,n2,n1) , YAKL_LAMBDA (int k, int j, int i) {
      arr3d(i,j,k) = 1;
    });
    if ( abs(sum(arr3d) - (double) n1*n2*n3) / (double) (n1*n2*n3) > 1.e-13) die("ERROR: Wrong sum for arr3d");

    yakl::memset(arr3d,0.);

    parallel_outer( "blah" , Bounds<1>(n1) , YAKL_LAMBDA (int k) {
      parallel_inner( Bounds<2>(n2,n3) , [&] (int j, int i) {
        arr3d(k,j,i) = 2.;
      });
      fence_inner();
      parallel_inner( Bounds<2>(n2,n3) , [&] (int j, int i) {
        arr3d(k,j,i) = 3.;
      });
      fence_inner();
      single_inner( [&] () {
        arr3d(k,1,1) = 0;
      });
    } , LaunchConfig<n2*n3>() );

    real exact = (double) n1*n2*n3*3 - (double) n1*3;
    if ( abs(sum(arr3d) - exact) / exact > 1.e-13) die("ERROR: Wrong sum for arr3d");
  }
  yakl::finalize();
  
  return 0;
}

