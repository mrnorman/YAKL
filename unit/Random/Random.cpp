
#include <iostream>
#include "YAKL.h"

using yakl::Array;
using yakl::styleC;
using yakl::memHost;
using yakl::memDevice;
using yakl::c::parallel_for;
using yakl::c::Bounds;
using yakl::c::SimpleBounds;
using yakl::COLON;

typedef double real;

typedef Array<real,1,memDevice,styleC> real1d;
typedef Array<real,2,memDevice,styleC> real2d;
typedef Array<real,1,memHost  ,styleC> realHost1d;

void die(std::string msg) {
  std::cerr << msg << std::endl;
  exit(-1);
}


int main() {
  yakl::init();
  {
    int constexpr n = 1024*1024;
    real1d arr("arr",n);
    auto clk = std::clock();
    parallel_for( n , YAKL_LAMBDA (int i) {
      yakl::Random rand(static_cast<unsigned long long>( clk ) + i);
      arr(i) = rand.genFP<real>( ) ;
    });
    // Compute mean
    real avg = yakl::intrinsics::sum(arr) / n;

    // Compute variance
    real1d varArr("varArr",n);
    parallel_for( n , YAKL_LAMBDA (int i) {
      real absdiff = abs(arr(i) - avg);
      varArr(i) = absdiff * absdiff;
    });
    real var = yakl::intrinsics::sum(varArr) / n;

    // Compute std dev
    real stddev = sqrt(var);

    // Compute skewness
    real1d skArr("skArr",n);
    parallel_for( n , YAKL_LAMBDA (int i) {
      real tmp = ( arr(i) - avg )  / stddev;
      skArr(i) = tmp*tmp*tmp;
    });
    real skew = yakl::intrinsics::sum(skArr) / n;

    real1d absDiffArr("absDiffArr",n-1);
    parallel_for( n-1 , YAKL_LAMBDA (int i) {
      absDiffArr(i) = abs( arr(i+1) - arr(i) );
    });
    real avgAbsDiff = yakl::intrinsics::sum(absDiffArr) / n;

    int constexpr nbins = 100;
    real2d bins("bins",nbins,n);
    parallel_for( Bounds<2>(nbins,n) , YAKL_LAMBDA (int b, int i) {
      real lo = (double) (b  ) / (double) nbins;
      real hi = (double) (b+1) / (double) nbins;
      bins(b,i) = (arr(i) >= lo && arr(i) <= hi) ? 1 : 0;
    });
    real maxBinErr = 0;
    for (int i=0; i < nbins; i++) {
      real binFrac = yakl::intrinsics::sum( bins.slice<1>(i,yakl::COLON) ) / n;
      maxBinErr = std::max( maxBinErr , abs( binFrac - 1./nbins ) );
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
  }
  yakl::finalize();
  
  return 0;
}

