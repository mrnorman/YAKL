
#include <iostream>
#include <utility>
#include "YAKL.h"

using yakl::Array;
using yakl::parallel_for;
using yakl::Bounds;
using yakl::SimpleBounds;
using yakl::COLON;

typedef double real;

typedef Array<real * ,yakl::DeviceSpace> real1d;
typedef Array<uint64_t **,yakl::DeviceSpace> uint64_2d;
typedef Array<int * ,yakl::DeviceSpace> int1d;

void die(std::string msg) {
  Kokkos::abort(msg.c_str());
}


int main() {
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");

    // Exact vectors and host/device agreement catch implementation errors that simple distribution moments cannot.
    {
      static_assert(std::is_trivially_copyable_v<yakl::Random>);

      yakl::Random reference(0,0);
      if (reference.gen() != 0x6627e8d5e169c58dULL || reference.gen() != 0xbc57ac4c9b00dbd8ULL ||
          reference.gen() != 0xf8e4cca45cb200dbULL || reference.gen() != 0xb1a574eb097eff67ULL) {
        die("ERROR: Random does not match the Philox4x32-10 reference sequence");
      }

      uint64_t constexpr seed = 8675309;
      uint64_t constexpr stream = 42;
      yakl::Random first(seed,stream);
      yakl::Random second(seed,stream);
      for (int i=0; i < 32; i++) {
        if (first.gen() != second.gen()) die("ERROR: identical Random seed and stream are not reproducible");
      }

      yakl::Random original(seed,stream);
      yakl::Random copied(original);
      for (int i=0; i < 32; i++) {
        if (original.gen() != copied.gen()) die("ERROR: Random copy constructor did not preserve state");
      }

      yakl::Random copyAssigned(1,1);
      copyAssigned = original;
      if (copyAssigned.gen() != original.gen()) die("ERROR: Random copy assignment did not preserve state");

      yakl::Random moveSource(seed,stream);
      yakl::Random moveExpected(seed,stream);
      yakl::Random moved(std::move(moveSource));
      if (moved.gen() != moveExpected.gen()) die("ERROR: Random move constructor did not preserve state");

      yakl::Random moveAssignSource(seed,stream + 1);
      yakl::Random moveAssignExpected(seed,stream + 1);
      yakl::Random moveAssigned(2,2);
      moveAssigned = std::move(moveAssignSource);
      if (moveAssigned.gen() != moveAssignExpected.gen()) die("ERROR: Random move assignment did not preserve state");

      yakl::Random reseeded(3,3);
      reseeded.set_seed(seed,stream);
      yakl::Random seeded(seed,stream);
      if (reseeded.gen() != seeded.gen()) die("ERROR: Random reseeding did not reset the sequence");

      yakl::Random otherSeed(seed + 1,stream);
      yakl::Random otherStream(seed,stream + 1);
      yakl::Random baseline(seed,stream);
      if (otherSeed.gen() == baseline.gen()) die("ERROR: Random seed does not affect the generated sequence");
      baseline.set_seed(seed,stream);
      if (otherStream.gen() == baseline.gen()) die("ERROR: Random stream ID does not affect the generated sequence");

      for (int i=0; i < 4096; i++) {
        double unit = seeded.genFP<double>();
        float ranged = seeded.genFP<float>(-2.5f,3.5f);
        if (unit < 0 || unit >= 1) die("ERROR: Random unit value is outside [0,1)");
        if (ranged < -2.5f || ranged >= 3.5f) die("ERROR: Random ranged value is outside [-2.5,3.5)");
      }
      if (seeded.genFP<double>(4.25,4.25) != 4.25) die("ERROR: Random equal-bound range did not return its bound");

      int constexpr nstreams = 64;
      int constexpr ndraws = 8;
      uint64_2d deviceValues("Random host device agreement",nstreams,ndraws);
      parallel_for( "Random host device agreement" , nstreams , KOKKOS_LAMBDA (int i) {
        yakl::Random random(seed,static_cast<uint64_t>(i));
        for (int draw = 0; draw < ndraws; draw++) deviceValues(i,draw) = random.gen();
      });
      auto hostValues = deviceValues.createHostCopy();
      for (int i=0; i < nstreams; i++) {
        yakl::Random hostRandom(seed,static_cast<uint64_t>(i));
        for (int draw = 0; draw < ndraws; draw++) {
          if (hostValues(i,draw) != hostRandom.gen()) die("ERROR: Random host and device sequences differ");
        }
      }
    }

    int constexpr n = 1024*1024;
    real1d arr("arr",n);
    uint64_t constexpr statistical_seed = 1368976481;
    parallel_for( "Kernel 1" , n , KOKKOS_LAMBDA (int i) {
      yakl::Random random(statistical_seed,static_cast<uint64_t>(i));
      arr(i) = random.genFP<real>();
    });
    // Compute mean
    real avg = yakl::intrinsics::sum(arr) / n;

    // Compute variance
    real1d varArr("varArr",n);
    parallel_for( "Kernel 2" , n , KOKKOS_LAMBDA (int i) {
      real absdiff = abs(arr(i) - avg);
      varArr(i) = absdiff * absdiff;
    });
    real var = yakl::intrinsics::sum(varArr) / n;

    // Compute std dev
    real stddev = sqrt(var);

    // Compute skewness
    real1d skArr("skArr",n);
    parallel_for( "Kernel 3" , n , KOKKOS_LAMBDA (int i) {
      real tmp = ( arr(i) - avg )  / stddev;
      skArr(i) = tmp*tmp*tmp;
    });
    real skew = yakl::intrinsics::sum(skArr) / n;

    real1d absDiffArr("absDiffArr",n-1);
    parallel_for( "Kernel 4" , n-1 , KOKKOS_LAMBDA (int i) {
      absDiffArr(i) = abs( arr(i+1) - arr(i) );
    });
    real avgAbsDiff = yakl::intrinsics::sum(absDiffArr) / (n-1);

    int constexpr nbins = 100;
    int1d bins("bins",nbins);
    bins = 0;
    parallel_for( "Kernel 5" , n , KOKKOS_LAMBDA (int i) {
      int bin = static_cast<int>(arr(i)*nbins);
      Kokkos::atomic_inc(&bins(bin));
    });
    auto binsHost = bins.createHostCopy();
    real maxBinErr = 0;
    for (int i=0; i < nbins; i++) {
      real binFrac = static_cast<real>(binsHost(i)) / n;
      maxBinErr = std::max( maxBinErr , std::abs( binFrac - 1./nbins ) );
    }
    
    std::cout << "Mean:          " << avg        << "\n";
    std::cout << "Variance:      " << var        << "\n";
    std::cout << "Skewness:      " << skew       << "\n";
    std::cout << "Mean Abs Diff: " << avgAbsDiff << "\n";
    std::cout << "Max Bin Err:   " << maxBinErr  << "\n";

    if (abs(avg-0.5)/0.5 > 0.01)           { die("ERROR: mean is wrong"); }
    if (abs(var-(1./12.))/(1./12.) > 0.01) { die("ERROR: variance is wrong"); }
    if (abs(skew) > 0.01)                  { die("ERROR: skewness is wrong"); }
    if (abs(avgAbsDiff-1./3.) > 0.01)      { die("ERROR: avg abs diff is wrong"); }
    if (maxBinErr > 0.01)                  { die("ERROR: max bin error is wrong"); }
    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize(); 
  
  return 0;
}
