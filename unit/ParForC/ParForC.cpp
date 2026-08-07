
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


int main(int argc, char **argv) {
  #ifdef HAVE_MPI
    MPI_Init(&argc,&argv);
  #endif
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");
    static_assert(yakl::LoopSpec::is_cstyle && !yakl::LoopSpec::is_fstyle);
    static_assert(!std::is_same_v<yakl::LoopSpec,yakl::LoopSpec_F>);
    static_assert(!std::is_constructible_v<yakl::Bounds<1>,yakl::LoopSpec_F>);
    yakl::LoopSpec const extentSpec(5);
    yakl::LoopSpec const arbitrarySpec(-4,4,2);
    if (extentSpec.l != 0 || extentSpec.u != 4 || extentSpec.s != 1 || extentSpec.index_range() != 5 ||
        arbitrarySpec.l != -4 || arbitrarySpec.u != 4 || arbitrarySpec.s != 2 || arbitrarySpec.index_range() != 5) {
      die("ERROR: C-style LoopSpec has incorrect bounds or trip count");
    }
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
    yakl::autotune::parallel_for( "autotune zero work" , SimpleBounds<2>(5,0) , KOKKOS_LAMBDA (int, int) {
      sentinel(0) = -1;
    });
    if (sum(sentinel) != 42 || !yakl::autotune::autotune_contexts.empty()) {
      die("ERROR: zero-work autotuned launch executed or created tuning state");
    }

    // Exercise runtime Cartesian tiles with partial edge tiles and dimensions
    // both smaller and larger than the tile size. Atomic increments detect
    // duplicate visits as well as missing points.
    Array<int ***,yakl::DeviceSpace> tiled("tiled",3,5,10);
    for (int tile : {1,2,4,8}) {
      tiled = 0;
      parallel_for( "C-style tiled" , SimpleBounds<3>(3,5,10) , KOKKOS_LAMBDA (int k, int j, int i) {
        Kokkos::atomic_add(&tiled(k,j,i),1);
      }, yakl::Config<128>{tile});
      auto tiledHost = tiled.createHostCopy();
      for (int k=0; k < 3; k++) {
        for (int j=0; j < 5; j++) {
          for (int i=0; i < 10; i++) {
            if (tiledHost(k,j,i) != 1) die("ERROR: C-style tiled launch did not visit every point exactly once");
          }
        }
      }
    }

    // A large tile on rank-eight bounds must iterate only valid points rather
    // than all tile^rank padded positions.
    Array<int *,yakl::DeviceSpace> rankEight("rank eight tiled",16);
    rankEight = 0;
    parallel_for( "rank eight tiled" , SimpleBounds<8>(1,2,1,2,1,2,1,2) ,
                  KOKKOS_LAMBDA (int, int i1, int, int i3, int, int i5, int, int i7) {
      int const linear = ((i1*2+i3)*2+i5)*2+i7;
      Kokkos::atomic_add(&rankEight(linear),1);
    }, yakl::Config<128>{8});
    auto rankEightHost = rankEight.createHostCopy();
    for (int i=0; i < 16; i++) {
      if (rankEightHost(i) != 1) die("ERROR: rank-eight tiled launch did not visit every point exactly once");
    }

    Array<int *,yakl::DeviceSpace> strided("strided",5);
    strided = 0;
    parallel_for( "strided inclusive endpoint" ,
                  Bounds<1>(yakl::LoopSpec(0,4,2)) , KOKKOS_LAMBDA (ptrdiff_t i) {
      strided(i) = i + 1;
    });
    if (sum(strided) != 9) die("ERROR: C-style strided launch omitted its final valid iteration");

    // Drive every launch-bound / tile-size combination through the autotuner,
    // including partial multidimensional edge tiles and the selected-config path.
    std::string const tuneLabel = "unit autotune";
    std::string const tuneKey = tuneLabel + ":3x5x7_iterations";
    yakl::autotune::autotune_contexts.erase(tuneKey);
    Array<int ***,yakl::DeviceSpace> tuned("tuned",3,5,7);
    tuned = 0;
    for (int iter=0; iter <= yakl::autotune::AutotuneContext::total_tests; iter++) {
      yakl::autotune::parallel_for( tuneLabel , SimpleBounds<3>(3,5,7) , KOKKOS_LAMBDA (int k, int j, int i) {
        Kokkos::atomic_add(&tuned(k,j,i),1);
      });
    }
    auto const &tuneContext = yakl::autotune::autotune_contexts.at(tuneKey);
    if (tuneContext.tests_performed != yakl::autotune::AutotuneContext::total_tests) {
      die("ERROR: autotuned C-style launch did not complete its state machine");
    }
    auto tunedHost = tuned.createHostCopy();
    for (int k=0; k < 3; k++) {
      for (int j=0; j < 5; j++) {
        for (int i=0; i < 7; i++) {
          if (tunedHost(k,j,i) != yakl::autotune::AutotuneContext::total_tests+1) {
            die("ERROR: autotuned tiled launch did not visit every point exactly once");
          }
        }
      }
    }
    std::array<bool,4> foundTiles = {false,false,false,false};
    for (int index=0; index < yakl::autotune::configuration_count; index++) {
      auto const [threads,tile] = yakl::autotune::get_config(index);
      (void) threads;
      for (int i=0; i < 4; i++) {
        foundTiles[i] = foundTiles[i] || tile == static_cast<int>(yakl::autotune::tile_sizes[i]);
      }
    }
    for (bool found : foundTiles) {
      if (!found) die("ERROR: autotuner did not include every requested tile size");
    }
    for (int index=0; index < yakl::autotune::configuration_count; index++) {
      if (tuneContext.sample_counts[index] != yakl::autotune::AutotuneContext::tests_per_config-1 ||
          !std::isfinite(tuneContext.timings[index]) || tuneContext.timings[index] < 0) {
        die("ERROR: completed autotune context has incorrect sample accounting");
      }
    }

    // Leave partial contexts alive through yakl::finalize(). This exercises warmup-only state, a measured partial
    // configuration, a completed first configuration, and the next configuration's warmup.
    std::array<int,4> const partialLaunchCounts = {1,2,5,6};
    for (int partial=0; partial < static_cast<int>(partialLaunchCounts.size()); partial++) {
      std::string const partialLabel = "unit partial autotune " + std::to_string(partial);
      std::string const partialKey = partialLabel + ":9_iterations";
      yakl::autotune::autotune_contexts.erase(partialKey);
      for (int launch=0; launch < partialLaunchCounts[partial]; launch++) {
        yakl::autotune::parallel_for( partialLabel , 9 , KOKKOS_LAMBDA (int) {} );
      }
      auto const &partialContext = yakl::autotune::autotune_contexts.at(partialKey);
      if (partialContext.tests_performed != partialLaunchCounts[partial]) {
        die("ERROR: partial autotune context has an incorrect launch count");
      }
      if (partialLaunchCounts[partial] == 1) {
        if (partialContext.best_index != -1 || partialContext.sample_counts[0] != 0) {
          die("ERROR: autotune warmup was treated as a measured result");
        }
      } else {
        if (partialContext.best_index != 0 || partialContext.sample_counts[0] != std::min(partialLaunchCounts[partial]-1,4)) {
          die("ERROR: partial autotune context did not retain its measured best configuration");
        }
      }
      if (partialLaunchCounts[partial] == 6 && partialContext.sample_counts[1] != 0) {
        die("ERROR: second-configuration warmup was treated as a measured result");
      }
    }

    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize(); 
  #ifdef HAVE_MPI
    MPI_Finalize();
  #endif
  
  return 0;
}
