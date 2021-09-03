
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
    int constexpr n1 = 1024;
    int constexpr n2 = 32;
    int constexpr n3 = 16;
    real1d arr1d("arr1d",n1);
    real3d arr3d("arr3d",n1,n2,n3);
    // Test with labels and SimpleBounds
    parallel_for( "mylabel" , n1 , YAKL_LAMBDA (int i) {
      arr1d(i) = 1;
    });
    if (yakl::intrinsics::sum(arr1d) != n1) die("ERROR: Wrong sum for arr1d");
    parallel_for( "mylabel" , SimpleBounds<3>(n3,n2,n1) , YAKL_LAMBDA (int k, int j, int i) {
      arr3d(i,j,k) = 1;
    });
    if (yakl::intrinsics::sum(arr3d) != n1*n2*n3) die("ERROR: Wrong sum for arr3d");
  }
  yakl::finalize();
  
  return 0;
}

