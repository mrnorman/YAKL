
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
typedef Array<real **,yakl::DeviceSpace> real2d;
typedef Array<real * ,Kokkos::HostSpace> realHost1d;

void die(std::string msg) {
  Kokkos::abort(msg.c_str());
}


int main() {
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");

    // Reproducibility, reseeding, and value bounds are more useful failures than
    // the statistical test below when the PRNG implementation changes.
    {
      yakl::Random default1;
      yakl::Random default2;
      for (int i=0; i < 32; i++) {
        if (default1.gen() != default2.gen()) { die("ERROR: default Random seed is not reproducible"); }
      }

      unsigned long long constexpr seed = 8675309;
      yakl::Random original(seed);
      yakl::Random copied(original);
      for (int i=0; i < 32; i++) {
        if (original.gen() != copied.gen()) { die("ERROR: Random copy constructor did not preserve state"); }
      }

      yakl::Random copyAssigned(1);
      copyAssigned = original;
      if (copyAssigned.gen() != original.gen()) { die("ERROR: Random copy assignment did not preserve state"); }

      yakl::Random moveSource(seed);
      yakl::Random moveExpected(seed);
      yakl::Random moved(std::move(moveSource));
      if (moved.gen() != moveExpected.gen()) { die("ERROR: Random move constructor did not preserve state"); }

      yakl::Random moveAssignSource(seed + 1);
      yakl::Random moveAssignExpected(seed + 1);
      yakl::Random moveAssigned(2);
      moveAssigned = std::move(moveAssignSource);
      if (moveAssigned.gen() != moveAssignExpected.gen()) { die("ERROR: Random move assignment did not preserve state"); }

      yakl::Random reseeded(3);
      reseeded.set_seed(seed);
      yakl::Random seeded(seed);
      if (reseeded.gen() != seeded.gen()) { die("ERROR: Random reseeding did not reset the sequence"); }

      for (int i=0; i < 1024; i++) {
        double unit = seeded.genFP<double>();
        float ranged = seeded.genFP<float>(-2.5f,3.5f);
        if (unit < 0 || unit > 1) { die("ERROR: Random unit value is outside [0,1]"); }
        if (ranged < -2.5f || ranged > 3.5f) { die("ERROR: Random ranged value is outside its bounds"); }
      }
      if (seeded.genFP<double>(4.25,4.25) != 4.25) { die("ERROR: Random equal-bound range did not return its bound"); }
    }

    int constexpr n = 1024*1024;
    real1d arr("arr",n);
    auto clk = std::clock();
    parallel_for( "Kernel 1" , n , KOKKOS_LAMBDA (int i) {
      yakl::Random rand(static_cast<unsigned long long>( clk ) + i);
      arr(i) = rand.genFP<real>( ) ;
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
    real avgAbsDiff = yakl::intrinsics::sum(absDiffArr) / n;

    int constexpr nbins = 100;
    real2d bins("bins",nbins,n);
    parallel_for( "Kernel 5" , Bounds<2>(nbins,n) , KOKKOS_LAMBDA (int b, int i) {
      real lo = (double) (b  ) / (double) nbins;
      real hi = (double) (b+1) / (double) nbins;
      bins(b,i) = (arr(i) >= lo && arr(i) <= hi) ? 1 : 0;
    });
    real maxBinErr = 0;
    for (int i=0; i < nbins; i++) {
      real binFrac = yakl::intrinsics::sum( bins.slice<1>(i,yakl::COLON) ) / n;
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
