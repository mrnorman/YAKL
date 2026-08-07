
#include <iostream>
#include <utility>
#include "YAKL.h"

using yakl::Array;
using yakl::parallel_for;

typedef double real;

typedef Array<real * ,yakl::DeviceSpace> real1d;
typedef Array<real **,yakl::DeviceSpace> real2d;
typedef Array<uint64_t **,yakl::DeviceSpace> uint64_2d;
typedef Array<int * ,yakl::DeviceSpace> int1d;

void die(std::string msg) {
  Kokkos::abort(msg.c_str());
}


void require_same_sequence(yakl::Random &first, yakl::Random &second, int draws, char const *message) {
  for (int i=0; i < draws; i++) {
    if (first.gen_uniform<uint64_t>() != second.gen_uniform<uint64_t>()) die(message);
  }
}


uint64_t next_after(uint64_t seed, uint64_t stream, int draws) {
  yakl::Random random(seed,stream);
  for (int i=0; i < draws; i++) (void) random.gen_uniform<uint64_t>();
  return random.gen_uniform<uint64_t>();
}


KOKKOS_INLINE_FUNCTION yakl::SArray<real,12> distribution_sample(yakl::Random &random) {
  yakl::SArray<real,12> values;
  values( 0) = random.gen_uniform();
  values( 1) = random.gen_uniform<double>(-2.,3.);
  values( 2) = random.gen_normal<double>();
  values( 3) = random.gen_normal<double>();
  values( 4) = random.gen_normal<double,false>(2.,0.5);
  values( 5) = random.gen_bernoulli(0.3) ? 1. : 0.;
  values( 6) = random.gen_exponential<double>(2.);
  values( 7) = random.gen_lognormal<double>(0.2,0.6);
  values( 8) = random.gen_normal();
  values( 9) = random.gen_exponential();
  values(10) = random.gen_lognormal();
  values(11) = random.gen_bernoulli() ? 1. : 0.;
  return values;
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
      static_assert(std::is_same_v<decltype(reference.gen_uniform()),float>);
      static_assert(std::is_same_v<decltype(reference.gen_normal()),float>);
      static_assert(std::is_same_v<decltype(reference.gen_exponential()),float>);
      static_assert(std::is_same_v<decltype(reference.gen_lognormal()),float>);
      if (reference.gen_uniform<uint64_t>() != 0x6627e8d5e169c58dULL ||
          reference.gen_uniform<uint64_t>() != 0xbc57ac4c9b00dbd8ULL ||
          reference.gen_uniform<uint64_t>() != 0xf8e4cca45cb200dbULL ||
          reference.gen_uniform<uint64_t>() != 0xb1a574eb097eff67ULL) {
        die("ERROR: Random does not match the Philox4x32-10 reference sequence");
      }

      uint64_t constexpr seed = 8675309;
      uint64_t constexpr stream = 42;
      yakl::Random first(seed,stream);
      yakl::Random second(seed,stream);
      require_same_sequence(first,second,32,"ERROR: identical Random seed and stream are not reproducible");

      yakl::Random original(seed,stream);
      yakl::Random copied(original);
      require_same_sequence(original,copied,32,"ERROR: Random copy constructor did not preserve state");

      yakl::Random copyAssigned(1,1);
      copyAssigned = original;
      require_same_sequence(copyAssigned,original,1,"ERROR: Random copy assignment did not preserve state");

      yakl::Random moveSource(seed,stream);
      yakl::Random moveExpected(seed,stream);
      yakl::Random moved(std::move(moveSource));
      require_same_sequence(moved,moveExpected,1,"ERROR: Random move constructor did not preserve state");

      yakl::Random moveAssignSource(seed,stream + 1);
      yakl::Random moveAssignExpected(seed,stream + 1);
      yakl::Random moveAssigned(2,2);
      moveAssigned = std::move(moveAssignSource);
      require_same_sequence(moveAssigned,moveAssignExpected,1,"ERROR: Random move assignment did not preserve state");

      yakl::Random reseeded(3,3);
      reseeded.set_seed(seed,stream);
      yakl::Random seeded(seed,stream);
      require_same_sequence(reseeded,seeded,1,"ERROR: Random reseeding did not reset the sequence");

      yakl::Random otherSeed(seed + 1,stream);
      yakl::Random otherStream(seed,stream + 1);
      yakl::Random baseline(seed,stream);
      if (otherSeed.gen_uniform<uint64_t>() == baseline.gen_uniform<uint64_t>()) {
        die("ERROR: Random seed does not affect the generated sequence");
      }
      baseline.set_seed(seed,stream);
      if (otherStream.gen_uniform<uint64_t>() == baseline.gen_uniform<uint64_t>()) {
        die("ERROR: Random stream ID does not affect the generated sequence");
      }

      for (int i=0; i < 4096; i++) {
        double unit = seeded.gen_uniform<double>();
        float ranged = seeded.gen_uniform<float>(-2.5f,3.5f);
        if (unit < 0 || unit >= 1) die("ERROR: Random unit value is outside [0,1)");
        if (ranged < -2.5f || ranged >= 3.5f) die("ERROR: Random ranged value is outside [-2.5,3.5)");
      }
      if (seeded.gen_uniform<double>(4.25,4.25) != 4.25) {
        die("ERROR: Random equal-bound range did not return its bound");
      }

      // A direct ub-lb calculation overflows for this finite interval and collapses almost every draw near +max.
      int constexpr nstreams = 64;
      double constexpr maximum = std::numeric_limits<double>::max();
      yakl::Random extremeHost(seed,stream);
      bool hostNegative = false;
      bool hostPositive = false;
      for (int draw=0; draw < 256; draw++) {
        double const value = extremeHost.gen_uniform<double>(-maximum,maximum);
        if (!(value >= -maximum && value < maximum)) die("ERROR: Random extreme host range produced an invalid value");
        hostNegative = hostNegative || value < 0;
        hostPositive = hostPositive || value > 0;
      }
      if (!hostNegative || !hostPositive) die("ERROR: Random extreme host range is severely biased");

      int1d extremeDeviceFlags("Random extreme device range",nstreams);
      parallel_for("Random extreme device range",nstreams,KOKKOS_LAMBDA (int i) {
        double constexpr localMaximum = std::numeric_limits<double>::max();
        yakl::Random random(seed,static_cast<uint64_t>(i));
        int flags = 0;
        for (int draw=0; draw < 256; draw++) {
          double const value = random.gen_uniform<double>(-localMaximum,localMaximum);
          if (!(value >= -localMaximum && value < localMaximum)) flags |= 1;
          if (value < 0) flags |= 2;
          if (value > 0) flags |= 4;
        }
        extremeDeviceFlags(i) = flags;
      });
      auto extremeHostFlags = extremeDeviceFlags.createHostCopy();
      for (int i=0; i < nstreams; i++) {
        if (extremeHostFlags(i) != 6) die("ERROR: Random extreme device range is invalid or severely biased");
      }

      int constexpr ndraws = 8;
      uint64_2d deviceValues("Random host device agreement",nstreams,ndraws);
      parallel_for( "Random host device agreement" , nstreams , KOKKOS_LAMBDA (int i) {
        yakl::Random random(seed,static_cast<uint64_t>(i));
        for (int draw = 0; draw < ndraws; draw++) deviceValues(i,draw) = random.gen_uniform<uint64_t>();
      });
      auto hostValues = deviceValues.createHostCopy();
      for (int i=0; i < nstreams; i++) {
        yakl::Random hostRandom(seed,static_cast<uint64_t>(i));
        for (int draw = 0; draw < ndraws; draw++) {
          if (hostValues(i,draw) != hostRandom.gen_uniform<uint64_t>()) {
            die("ERROR: Random host and device sequences differ");
          }
        }
      }

      // Two cached normal calls consume exactly one Box-Muller pair, while disabling the cache consumes one pair per call.
      yakl::Random cachedNormal(seed,stream);
      (void) cachedNormal.gen_normal<double>();
      (void) cachedNormal.gen_normal<double>();
      if (cachedNormal.gen_uniform<uint64_t>() != next_after(seed,stream,2)) {
        die("ERROR: cached normal generation consumed an incorrect number of uniform values");
      }

      yakl::Random uncachedNormal(seed,stream);
      (void) uncachedNormal.gen_normal<double,false>();
      (void) uncachedNormal.gen_normal<double,false>();
      if (uncachedNormal.gen_uniform<uint64_t>() != next_after(seed,stream,4)) {
        die("ERROR: uncached normal generation consumed an incorrect number of uniform values");
      }

      yakl::Random spareSource(seed,stream);
      (void) spareSource.gen_normal<double>();
      yakl::Random spareCopy(spareSource);
      if (spareSource.gen_normal<double>() != spareCopy.gen_normal<double>()) {
        die("ERROR: Random copy did not preserve the cached normal value");
      }
      spareSource.set_seed(seed,stream);
      yakl::Random normalReseedExpected(seed,stream);
      if (spareSource.gen_normal<double>() != normalReseedExpected.gen_normal<double>()) {
        die("ERROR: Random reseeding did not clear the cached normal value");
      }

      yakl::Random endpointBernoulli(seed,stream);
      if (endpointBernoulli.gen_bernoulli(0.) || !endpointBernoulli.gen_bernoulli(1.)) {
        die("ERROR: Bernoulli endpoint probabilities are incorrect");
      }
      yakl::Random endpointExpected(seed,stream);
      if (endpointBernoulli.gen_uniform<uint64_t>() != endpointExpected.gen_uniform<uint64_t>()) {
        die("ERROR: Bernoulli endpoint probabilities consumed randomness");
      }

      yakl::Random zeroStddev(seed,stream);
      if (zeroStddev.gen_normal<double>(4.25,0.) != 4.25 || zeroStddev.gen_lognormal<double>(0.,0.) != 1.) {
        die("ERROR: zero-deviation normal or lognormal result is incorrect");
      }
      yakl::Random zeroStddevExpected(seed,stream);
      if (zeroStddev.gen_uniform<uint64_t>() != zeroStddevExpected.gen_uniform<uint64_t>()) {
        die("ERROR: zero-deviation normal or lognormal generation consumed randomness");
      }

      int constexpr ndistributions = 12;
      real2d deviceDistributions("Random distribution host device agreement",nstreams,ndistributions);
      parallel_for("Random distribution host device agreement",nstreams,KOKKOS_LAMBDA (int i) {
        yakl::Random random(seed,static_cast<uint64_t>(i));
        auto const sample = distribution_sample(random);
        for (int j=0; j < ndistributions; j++) deviceDistributions(i,j) = sample(j);
      });
      auto hostDistributions = deviceDistributions.createHostCopy();
      for (int i=0; i < nstreams; i++) {
        yakl::Random random(seed,static_cast<uint64_t>(i));
        auto const expected = distribution_sample(random);
        for (int j=0; j < ndistributions; j++) {
          real const relativeTolerance = j >= 8 && j <= 10 ? 2.e-6 : 1.e-12;
          real const tolerance = relativeTolerance*(1+std::abs(expected(j)));
          if (std::abs(hostDistributions(i,j)-expected(j)) > tolerance) {
            die("ERROR: Random distribution differs between host and device");
          }
        }
      }
    }

    int constexpr n = 1024*1024;
    real1d arr("arr",n);
    uint64_t constexpr statistical_seed = 1368976481;
    parallel_for( "Kernel 1" , n , KOKKOS_LAMBDA (int i) {
      yakl::Random random(statistical_seed,static_cast<uint64_t>(i));
      arr(i) = random.gen_uniform<real>();
    });
    real avg = yakl::intrinsics::sum(arr) / n;

    // Reuse one full-size scratch allocation for the variance, skewness, and serial-correlation checks.
    real1d scratch("scratch",n);
    parallel_for( "Kernel 2" , n , KOKKOS_LAMBDA (int i) {
      real absdiff = abs(arr(i) - avg);
      scratch(i) = absdiff * absdiff;
    });
    real var = yakl::intrinsics::sum(scratch) / n;
    real stddev = sqrt(var);

    parallel_for( "Kernel 3" , n , KOKKOS_LAMBDA (int i) {
      real tmp = ( arr(i) - avg )  / stddev;
      scratch(i) = tmp*tmp*tmp;
    });
    real skew = yakl::intrinsics::sum(scratch) / n;

    parallel_for( "Kernel 4" , n , KOKKOS_LAMBDA (int i) {
      scratch(i) = i < n-1 ? abs(arr(i+1)-arr(i)) : 0;
    });
    real avgAbsDiff = yakl::intrinsics::sum(scratch) / (n-1);

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

    // Stress all transformed distributions with independent per-element streams. Both members of cached
    // Box-Muller pairs and the uncached path are represented in the normal statistics.
    real1d normalValues("normalValues",n);
    real1d uncachedNormalValues("uncachedNormalValues",n);
    real1d exponentialValues("exponentialValues",n);
    real1d lognormalValues("lognormalValues",n);
    real1d bernoulliValues("bernoulliValues",n);
    parallel_for("Random transformed distributions",n,KOKKOS_LAMBDA (int i) {
      yakl::Random random(statistical_seed+1,static_cast<uint64_t>(i));
      real const normal0 = random.gen_normal<real>();
      real const normal1 = random.gen_normal<real>();
      normalValues(i) = i%2 == 0 ? normal0 : normal1;
      uncachedNormalValues(i) = random.gen_normal<real,false>();
      exponentialValues(i) = random.gen_exponential<real>(2.);
      lognormalValues(i) = random.gen_lognormal<real>(0.2,0.6);
      bernoulliValues(i) = random.gen_bernoulli(0.37) ? 1. : 0.;
    });

    auto moments = [&] (real1d const & values) {
      real const mean = yakl::intrinsics::sum(values) / n;
      auto const valuesLocal = values;
      parallel_for("Random distribution second moment",n,KOKKOS_LAMBDA (int i) {
        scratch(i) = valuesLocal(i)*valuesLocal(i);
      });
      real const secondMoment = yakl::intrinsics::sum(scratch) / n;
      return std::pair<real,real>{mean,secondMoment-mean*mean};
    };

    auto const normalMoments = moments(normalValues);
    auto const uncachedNormalMoments = moments(uncachedNormalValues);
    auto const exponentialMoments = moments(exponentialValues);
    auto const lognormalMoments = moments(lognormalValues);
    real const bernoulliMean = yakl::intrinsics::sum(bernoulliValues) / n;
    real constexpr exponentialMean = 0.5;
    real constexpr exponentialVariance = 0.25;
    real const lognormalMean = std::exp(0.2 + 0.6*0.6/2);
    real const lognormalVariance = (std::exp(0.6*0.6)-1)*std::exp(2*0.2+0.6*0.6);

    std::cout << "Normal mean / variance:          " << normalMoments.first << " / " << normalMoments.second << "\n";
    std::cout << "Uncached normal mean / variance: " << uncachedNormalMoments.first << " / "
              << uncachedNormalMoments.second << "\n";
    std::cout << "Exponential mean / variance:     " << exponentialMoments.first << " / "
              << exponentialMoments.second << "\n";
    std::cout << "Lognormal mean / variance:       " << lognormalMoments.first << " / "
              << lognormalMoments.second << "\n";
    std::cout << "Bernoulli mean:                  " << bernoulliMean << "\n";

    if (std::abs(normalMoments.first) > 0.01 || std::abs(normalMoments.second-1) > 0.02) {
      die("ERROR: cached normal distribution moments are incorrect");
    }
    if (std::abs(uncachedNormalMoments.first) > 0.01 || std::abs(uncachedNormalMoments.second-1) > 0.02) {
      die("ERROR: uncached normal distribution moments are incorrect");
    }
    if (std::abs(exponentialMoments.first-exponentialMean)/exponentialMean > 0.02 ||
        std::abs(exponentialMoments.second-exponentialVariance)/exponentialVariance > 0.03) {
      die("ERROR: exponential distribution moments are incorrect");
    }
    if (std::abs(lognormalMoments.first-lognormalMean)/lognormalMean > 0.02 ||
        std::abs(lognormalMoments.second-lognormalVariance)/lognormalVariance > 0.04) {
      die("ERROR: lognormal distribution moments are incorrect");
    }
    if (std::abs(bernoulliMean-0.37) > 0.005) die("ERROR: Bernoulli distribution mean is incorrect");
    if (yakl::intrinsics::minval(exponentialValues) < 0 || yakl::intrinsics::minval(lognormalValues) <= 0) {
      die("ERROR: exponential or lognormal distribution violated its support");
    }
    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize(); 
  
  return 0;
}
