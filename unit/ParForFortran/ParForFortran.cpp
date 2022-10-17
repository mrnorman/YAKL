
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
using yakl::InnerHandler;
using yakl::fortran::parallel_for;
using yakl::fortran::Bounds;
using yakl::fortran::SimpleBounds;
using yakl::COLON;
using yakl::fence_inner;
using yakl::intrinsics::sum;

typedef double real;

typedef Array<real,1,memDevice,styleFortran> real1d;
typedef Array<real,3,memDevice,styleFortran> real3d;


void die(std::string msg) {
  yakl::yakl_throw(msg.c_str());
}


int main() {
  yakl::init();
  {
    int constexpr n1 = 4;
    int constexpr n2 = 16;
    int constexpr n3 = 128;
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

    parallel_outer( "blah" , Bounds<1>(n3) , YAKL_LAMBDA (int k, InnerHandler handler ) {
      parallel_inner( Bounds<2>(n2,n1) , [&] (int j, int i) {
        arr3d(i,j,k) = 2.;
      } , handler );
      fence_inner( handler );
      parallel_inner( Bounds<2>(n2,n1) , [&] (int j, int i) {
        arr3d(i,j,k) = 3.;
      } , handler );
      fence_inner( handler );
      single_inner( [&] () {
        arr3d(1,1,k) = 0;
      } , handler );
    } , LaunchConfig<n2*n1>() );

    real exact = (double) n1*n2*n3*3 - (double) n3*3;
    if ( abs(sum(arr3d) - exact) / exact > 1.e-13) die("ERROR: Wrong sum for arr3d");

    #ifdef YAKL_ARCH_OPENMP
    {
      int constexpr nz = 8;
      int constexpr nx = 8;
      yakl::ScalarLiveOut<int> tot_outer(0);
      yakl::ScalarLiveOut<int> tot      (0);
      parallel_outer( nz , YAKL_LAMBDA (int k, InnerHandler handler) {
        yakl::atomicAdd( tot_outer() , omp_get_thread_num() );
        parallel_inner( nx , [&] (int i) {
          yakl::atomicAdd( tot() , omp_get_thread_num() );
        } , handler );
      } , LaunchConfig<nz>() );
      if (tot_outer.hostRead() != 28 ) yakl::yakl_throw("ERROR: Wrong tot_outer");
      if (tot      .hostRead() != 224) yakl::yakl_throw("ERROR: Wrong tot");
    }
    #endif
  }
  yakl::finalize();
  
  return 0;
}

