
#include <iostream>
#include "YAKL.h"

using yakl::Array;
using yakl::parallel_for;
using yakl::Bounds;
using yakl::SimpleBounds;
using yakl::COLON;

typedef double real;

typedef Array<real *,Kokkos::HostSpace> realHost1d;
typedef Array<real *,yakl::DeviceSpace> real1d;


void die(std::string msg) {
  Kokkos::abort(msg.c_str());
}


int main(int argc, char **argv) {
  #ifdef HAVE_MPI
    MPI_Init(&argc,&argv);
  #endif
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");
    int constexpr n = 1024*1024 + 1;
    real1d data("data",n);
    parallel_for( "Initialize data" , n , KOKKOS_LAMBDA (int i) {
      data(i) = i - (n-1)/2.;
    });
    real sum = yakl::intrinsics::sum   ( data );
    real min = yakl::intrinsics::minval( data );
    real max = yakl::intrinsics::maxval( data );
    if ( abs(sum) > 1.e-13 ) { die("ERROR: Wrong device sum"); }
    if ( abs(min + (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device min"); }
    if ( abs(max - (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device max"); }

    auto dataHost = data.createHostCopy();
    sum = yakl::intrinsics::sum   ( dataHost );
    min = yakl::intrinsics::minval( dataHost );
    max = yakl::intrinsics::maxval( dataHost );
    if ( abs(sum) > 1.e-13 ) { die("ERROR: Wrong device sum"); }
    if ( abs(min + (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device min"); }
    if ( abs(max - (n-1)/2.) > 1.e-13 ) { die("ERROR: Wrong device max"); }

    // Singleton inputs catch incorrect reducer initialization, while an
    // asymmetric all-negative input catches zero-initialized max reductions.
    real1d singleton("singleton",1);
    parallel_for( "Initialize singleton" , 1 , KOKKOS_LAMBDA (int) {
      singleton(0) = -17.25;
    });
    if (yakl::intrinsics::sum(singleton) != -17.25) { die("ERROR: Wrong singleton device sum"); }
    if (yakl::intrinsics::minval(singleton) != -17.25) { die("ERROR: Wrong singleton device min"); }
    if (yakl::intrinsics::maxval(singleton) != -17.25) { die("ERROR: Wrong singleton device max"); }

    int constexpr nneg = 7;
    real1d negative("negative",nneg);
    parallel_for( "Initialize negative data" , nneg , KOKKOS_LAMBDA (int i) {
      negative(i) = -2*i - 1;
    });
    if (yakl::intrinsics::sum(negative) != -49) { die("ERROR: Wrong all-negative device sum"); }
    if (yakl::intrinsics::minval(negative) != -13) { die("ERROR: Wrong all-negative device min"); }
    if (yakl::intrinsics::maxval(negative) != -1) { die("ERROR: Wrong all-negative device max"); }

    auto negativeHost = negative.createHostCopy();
    if (yakl::intrinsics::sum(negativeHost) != -49) { die("ERROR: Wrong all-negative host sum"); }
    if (yakl::intrinsics::minval(negativeHost) != -13) { die("ERROR: Wrong all-negative host min"); }
    if (yakl::intrinsics::maxval(negativeHost) != -1) { die("ERROR: Wrong all-negative host max"); }
    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize(); 
  #ifdef HAVE_MPI
    MPI_Finalize();
  #endif

  return 0;
}
