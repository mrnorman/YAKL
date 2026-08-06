
#include <iostream>
#include "YAKL.h"

/////////////////////////////////////////////////////////////////////
// Most of YAKL_parallel_for_c.h is tested in the CArray unit test.
// This is to cover what was not covered in CArray
/////////////////////////////////////////////////////////////////////

using yakl::Array;
using yakl::parallel_for;
using yakl::Bounds;
using yakl::SimpleBounds;
using yakl::COLON;
using yakl::intrinsics::sum;

typedef double real;

typedef Array<real *  ,yakl::DeviceSpace> real1d;
typedef Array<real ***,yakl::DeviceSpace> real3d;


void die(std::string msg) {
  Kokkos::abort(msg.c_str());
}


int main() {
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");
    int constexpr n1 = 128;
    int constexpr n2 = 16;
    int constexpr n3 = 4;
    real1d arr1d("arr1d",n1);
    real3d arr3d("arr3d",n1,n2,n3);
    // Test with labels and SimpleBounds
    parallel_for( "mylabel" , n1 , KOKKOS_LAMBDA (int i) {
      arr1d(i) = 1;
    });
    if ( abs(sum(arr1d) - n1) / (n1) > 1.e-13) die("ERROR: Wrong sum for arr1d");
    parallel_for( "mylabel" , SimpleBounds<3>(n1,n2,n3) , KOKKOS_LAMBDA (int k, int j, int i) {
      arr3d(k,j,i) = 1;
    });
    if ( abs(sum(arr3d) - (double) n1*n2*n3) / (double) (n1*n2*n3) > 1.e-13) die("ERROR: Wrong sum for arr3d");

    arr3d = 0.;

    // Zero-work launches must return without touching captures.
    Array<int *,yakl::DeviceSpace> sentinel("sentinel",1);
    sentinel = 42;
    parallel_for( "zero work" , SimpleBounds<2>(0,7) , KOKKOS_LAMBDA (int, int) {
      sentinel(0) = -1;
    });
    if (sum(sentinel) != 42) die("ERROR: zero-work C-style launch executed its kernel");

    // A strip size that does not divide the iteration count exercises the
    // guarded tail in the strip-mined launcher.
    int constexpr nstrip = 17;
    Array<int *,yakl::DeviceSpace> stripped("stripped",nstrip);
    stripped = 0;
    parallel_for( "strip tail" , nstrip , KOKKOS_LAMBDA (int i) {
      stripped(i) = i + 1;
    }, yakl::Config<128,4>{});
    if (sum(stripped) != nstrip*(nstrip+1)/2) die("ERROR: C-style strip-mined launch missed its tail");

    // Drive the autotuner through first-use, sampling, and selected-config
    // paths with a single-element kernel to keep the test inexpensive.
    std::string const tuneLabel = "unit autotune";
    std::string const tuneKey = tuneLabel + ":1_iterations";
    yakl::autotune::autotune_contexts.erase(tuneKey);
    Array<int *,yakl::DeviceSpace> tuned("tuned",1);
    tuned = 0;
    for (int iter=0; iter <= yakl::autotune::AutotuneContext::total_tests; iter++) {
      yakl::autotune::parallel_for( tuneLabel , 1 , KOKKOS_LAMBDA (int) {
        tuned(0)++;
      });
    }
    auto const &tuneContext = yakl::autotune::autotune_contexts.at(tuneKey);
    if (tuneContext.tests_performed != yakl::autotune::AutotuneContext::total_tests ||
        sum(tuned) != yakl::autotune::AutotuneContext::total_tests + 1) {
      die("ERROR: autotuned C-style launch did not complete its state machine");
    }
    if (yakl::autotune::get_config(-1) != std::make_pair(0,0)) die("ERROR: invalid autotune config lookup is incorrect");
    yakl::autotune::autotune_contexts.erase(tuneKey);

    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize(); 
  
  return 0;
}
