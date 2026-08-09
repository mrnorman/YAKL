
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


int main(int argc, char **argv) {
  #ifdef HAVE_MPI
    MPI_Init(&argc,&argv);
  #endif
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");
    static_assert(!yakl::LoopSpec_F::is_cstyle && yakl::LoopSpec_F::is_fstyle);
    static_assert(!std::is_same_v<yakl::LoopSpec,yakl::LoopSpec_F>);
    static_assert(!std::is_constructible_v<yakl::Bounds_F<1>,yakl::LoopSpec>);
    yakl::LoopSpec_F const extentSpec(5);
    yakl::LoopSpec_F const arbitrarySpec(-4,4,2);
    if (extentSpec.l != 1 || extentSpec.u != 5 || extentSpec.s != 1 || extentSpec.index_range() != 5 ||
        arbitrarySpec.l != -4 || arbitrarySpec.u != 4 || arbitrarySpec.s != 2 || arbitrarySpec.index_range() != 5) {
      die("ERROR: Fortran-style LoopSpec_F has incorrect bounds or trip count");
    }
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

    Array_F<int *,yakl::DeviceSpace> sentinel("sentinel",1);
    sentinel = 42;
    parallel_for_F( "zero work" , SimpleBounds_F<2>(0,7) , KOKKOS_LAMBDA (int, int) {
      sentinel(1) = -1;
    });
    if (sum(sentinel) != 42) die("ERROR: zero-work Fortran-style launch executed its kernel");

    // Arbitrary and negative lower bounds, non-unit strides, and independent partial edge tiles must all map back to the
    // correct Fortran indices.
    Array_F<int ***,yakl::DeviceSpace> tiled("Fortran tiled",{-5,3},{7,13},{-2,8});
    std::array<std::array<int,3>,4> const tileConfigs = {{{1,1,1},{1,2,4},{2,4,8},{8,3,2}}};
    for (auto const & tiles : tileConfigs) {
      tiled = 0;
      parallel_for_F( "Fortran-style tiled" ,
                      Bounds_F<3>({-5,3,2},{7,13,3},{-2,8,5}) , KOKKOS_LAMBDA (ptrdiff_t i, ptrdiff_t j, ptrdiff_t k) {
        Kokkos::atomic_add(&tiled(i,j,k),1);
      }, yakl::Config<128>{tiles[0],tiles[1],tiles[2]});
      auto tiledHost = tiled.createHostCopy();
      for (int k=-2; k <= 8; k++) {
        for (int j=7; j <= 13; j++) {
          for (int i=-5; i <= 3; i++) {
            bool const visited = (i+5)%2 == 0 && (j-7)%3 == 0 && (k+2)%5 == 0;
            if (tiledHost(i,j,k) != (visited ? 1 : 0)) {
              die("ERROR: Fortran-style tiled launch mapped arbitrary bounds incorrectly");
            }
          }
        }
      }
    }

    Array_F<int *,yakl::DeviceSpace> strided("strided",{-2,4});
    strided = 0;
    parallel_for_F( "strided inclusive endpoint" ,
                    Bounds_F<1>(yakl::LoopSpec_F(-2,4,3)) , KOKKOS_LAMBDA (ptrdiff_t i) {
      strided(i) = i + 3;
    });
    if (sum(strided) != 12) die("ERROR: Fortran-style strided launch omitted its final valid iteration");
    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize(); 
  #ifdef HAVE_MPI
    MPI_Finalize();
  #endif

  return 0;
}
