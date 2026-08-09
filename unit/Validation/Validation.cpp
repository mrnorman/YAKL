#include <iostream>
#include <limits>
#include <string>
#include "YAKL.h"

using yakl::Array;
using yakl::Array_F;
using yakl::Bnds;
using yakl::SArray_F;

static_assert(yakl::Config<128>::Thr == 128);
static_assert(sizeof(yakl::index_t)*8 == yakl::index_bits);
static_assert(sizeof(yakl::uindex_t)*8 == yakl::index_bits);
static_assert(std::is_signed_v<yakl::index_t>);
static_assert(std::is_unsigned_v<yakl::uindex_t>);

void fail(std::string const &message) {
  Kokkos::abort(message.c_str());
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  std::string const scenario = argv[1];
  #ifdef HAVE_MPI
    MPI_Init(&argc,&argv);
  #endif
  Kokkos::initialize();

  if (scenario == "allocate_before_init") {
    (void) yakl::alloc_device(16,"outside YAKL lifetime");
    return 0;
  } else if (scenario == "free_before_init") {
    void * ptr = Kokkos::kokkos_malloc("outside YAKL lifetime",16);
    yakl::free_device(ptr,"outside YAKL lifetime");
    return 0;
  } else if (scenario == "environment_negative") {
    setenv("GATOR_DISABLE","1",1);
    setenv("GATOR_INITIAL_MB","-17",1);
    setenv("GATOR_BLOCK_BYTES","-64",1);
    std::ostringstream warnings;
    auto *oldBuffer = std::cout.rdbuf(warnings.rdbuf());
    yakl::init();
    std::cout.rdbuf(oldBuffer);
    if (yakl::get_yakl_instance().use_pool()) fail("GATOR_DISABLE did not disable the pool");
    if (warnings.str().find("Defaulting to 4GB") == std::string::npos) {
      fail("negative GATOR_INITIAL_MB was accepted or emitted the wrong default diagnostic");
    }
    if (warnings.str().find("Defaulting to 4096 bytes") == std::string::npos) {
      fail("negative GATOR_BLOCK_BYTES was accepted or emitted the wrong default diagnostic");
    }
    yakl::finalize();
    Kokkos::finalize();
    #ifdef HAVE_MPI
      MPI_Finalize();
    #endif
    return 0;
  } else if (scenario == "config_disable") {
    unsetenv("GATOR_DISABLE");
    setenv("GATOR_INITIAL_MB","1",1);
    yakl::init(yakl::InitConfig().set_pool_enabled(false));
    if (yakl::get_yakl_instance().use_pool()) fail("InitConfig::set_pool_enabled(false) did not disable the pool");
    yakl::finalize();
    Kokkos::finalize();
    #ifdef HAVE_MPI
      MPI_Finalize();
    #endif
    return 0;
  } else if (scenario == "config_enable") {
    setenv("GATOR_DISABLE","1",1);
    setenv("GATOR_INITIAL_MB","1",1);
    yakl::init(yakl::InitConfig().set_pool_enabled(true));
    if (! yakl::get_yakl_instance().use_pool()) fail("explicit pool enable did not override GATOR_DISABLE");
    yakl::finalize();
    Kokkos::finalize();
    #ifdef HAVE_MPI
      MPI_Finalize();
    #endif
    return 0;
  } else if (scenario == "config_size_default") {
    unsetenv("GATOR_DISABLE");
    yakl::init(yakl::InitConfig().set_pool_size_mb(1));
    if (! yakl::get_yakl_instance().use_pool()) fail("setting a pool size unexpectedly disabled the pool");
    yakl::finalize();
    Kokkos::finalize();
    #ifdef HAVE_MPI
      MPI_Finalize();
    #endif
    return 0;
  } else if (scenario == "config_block_precedence") {
    size_t constexpr alignment = yakl::LinearAllocator::requiredAlignment;
    setenv("GATOR_INITIAL_MB","1",1);
    std::string const environment_block = std::to_string(3*alignment);
    setenv("GATOR_BLOCK_BYTES",environment_block.c_str(),1);
    yakl::init(yakl::InitConfig().set_pool_block_bytes(2*alignment));
    if (yakl::get_yakl_instance().pool.blockSize != 2*alignment) {
      fail("explicit pool block size did not override GATOR_BLOCK_BYTES");
    }
    yakl::finalize();
    Kokkos::finalize();
    #ifdef HAVE_MPI
      MPI_Finalize();
    #endif
    return 0;
  } else if (scenario == "lifecycle_reset") {
    yakl::init(yakl::InitConfig().set_pool_enabled(false));
    yakl::autotune::autotune_contexts["lifecycle reset"] = yakl::autotune::AutotuneContext();
    yakl::get_yakl_instance().timer.start("lifecycle reset");
    yakl::get_yakl_instance().timer.stop ("lifecycle reset");
    yakl::finalize();
    if (!yakl::autotune::autotune_contexts.empty() || !yakl::get_yakl_instance().timer.timers.empty() ||
        !yakl::get_yakl_instance().timer.active_stacks.empty()) {
      fail("yakl::finalize did not clear process-local autotune and timer state");
    }
    yakl::init(yakl::InitConfig().set_pool_enabled(false));
    if (!yakl::autotune::autotune_contexts.empty() || !yakl::get_yakl_instance().timer.timers.empty()) {
      fail("a new YAKL lifecycle interval inherited state from the previous interval");
    }
    yakl::finalize();
    Kokkos::finalize();
    #ifdef HAVE_MPI
      MPI_Finalize();
    #endif
    return 0;
  }

  yakl::init();

  if (scenario == "positive") {
    Array_F<int **,Kokkos::HostSpace> arr("arr",{-11,-9},{20,23});
    for (int j=20; j <= 23; j++) {
      for (int i=-11; i <= -9; i++) arr(i,j) = (i+11) + 3*(j-20);
    }
    for (size_t linear=0; linear < arr.size(); linear++) {
      auto index = arr.unpack_global_index(linear);
      if (index(1) < -11 || index(1) > -9 || index(2) < 20 || index(2) > 23 ||
          arr(index(1),index(2)) != arr.data()[linear]) {
        fail("Array_F arbitrary-lower-bound index validation failed");
      }
    }

    #if YAKL_INDEX_BITS == 64
    SArray_F<int,Bnds{-10000000000LL,-9999999998LL},Bnds{-7,-4}> stack;
    for (yakl::uindex_t linear=0; linear < stack.size(); linear++) {
      stack.data()[linear] = static_cast<int>(linear);
      auto index = stack.unpack_global_index(linear);
      if (stack(index(1),index(2)) != stack.data()[linear]) {
        fail("SArray_F large negative lower-bound validation failed");
      }
    }
    #endif

    yakl::Bounds_F<2> bounds({-7,5,3},{11,20,4});
    if (bounds.nIter != 15) fail("strided Bounds iteration count is incorrect");
    bool foundFinalPair = false;
    for (yakl::uindex_t linear=0; linear < bounds.nIter; linear++) {
      yakl::index_t i, j;
      bounds.unpack(linear,i,j);
      if (i < -7 || i > 5 || (i+7)%3 != 0 || j < 11 || j > 20 || (j-11)%4 != 0) {
        fail("strided Bounds unpack returned an invalid index");
      }
      foundFinalPair = foundFinalPair || (i == 5 && j == 19);
    }
    if (!foundFinalPair) fail("strided Bounds omitted its final valid iteration");

    yakl::index_t constexpr lowest  = std::numeric_limits<yakl::index_t>::min();
    yakl::index_t constexpr highest = std::numeric_limits<yakl::index_t>::max();
    yakl::Bounds_F<1> extreme(yakl::LoopSpec_F(lowest,highest-1,highest));
    yakl::index_t expected[3] = {lowest,-1,highest-1};
    for (yakl::uindex_t linear=0; linear < extreme.nIter; linear++) {
      yakl::index_t index;
      extreme.unpack(linear,index);
      if (index != expected[linear]) fail("extreme strided Bounds host unpack is incorrect");
    }
    Array<yakl::index_t *,yakl::DeviceSpace> device_indices("extreme strided indices",3);
    device_indices = 0;
    yakl::parallel_for_F("extreme strided Bounds",extreme,KOKKOS_LAMBDA (yakl::index_t index) {
      if      (index == lowest   ) device_indices(0) = index;
      else if (index == -1       ) device_indices(1) = index;
      else if (index == highest-1) device_indices(2) = index;
    });
    auto host_indices = device_indices.createHostCopy();
    for (size_t i=0; i < 3; i++) {
      if (host_indices(i) != expected[i]) fail("extreme strided Bounds device unpack is incorrect");
    }
  } else if (scenario == "array_slice") {
    Array<int **,Kokkos::HostSpace> arr("arr",2,3);
    auto slice = arr.slice<1>(2,yakl::COLON);
    (void) slice;
  } else if (scenario == "array_subset") {
    Array_F<int **,Kokkos::HostSpace> arr("arr",{-2,1},{7,9});
    auto subset = arr.subset_slowest_dimension(1,6);
    (void) subset;
  } else if (scenario == "array_reshape") {
    Array<int *,Kokkos::HostSpace> arr("arr",6);
    auto reshaped = arr.reshape(2,4);
    (void) reshaped;
  } else if (scenario == "array_f_bound_overflow") {
    using Bound = Array_F<int *,Kokkos::HostSpace>::AB;
    uintmax_t constexpr highest = std::numeric_limits<uintmax_t>::max();
    (void) Bound(highest-1,highest);
  } else if (scenario == "component_shape") {
    using namespace yakl::componentwise;
    Array<int **,Kokkos::HostSpace> left("left",2,3);
    Array<int **,Kokkos::HostSpace> right("right",2,2);
    auto result = left + right;
    (void) result;
  } else if (scenario == "sign_shape") {
    Array<int **,Kokkos::HostSpace> magnitude("magnitude",2,3);
    Array<int **,Kokkos::HostSpace> signSource("sign source",2,2);
    auto result = yakl::intrinsics::sign(magnitude,signSource);
    (void) result;
  } else if (scenario == "merge_value_shape") {
    Array<int **,Kokkos::HostSpace> trueValues("true values",2,3);
    Array<int **,Kokkos::HostSpace> falseValues("false values",2,2);
    Array<int **,Kokkos::HostSpace> condition("condition",2,3);
    auto result = yakl::intrinsics::merge(trueValues,falseValues,condition);
    (void) result;
  } else if (scenario == "merge_condition_shape") {
    Array<int **,Kokkos::HostSpace> trueValues("true values",2,3);
    Array<int **,Kokkos::HostSpace> falseValues("false values",2,3);
    Array<int **,Kokkos::HostSpace> condition("condition",2,2);
    auto result = yakl::intrinsics::merge(trueValues,falseValues,condition);
    (void) result;
  } else if (scenario == "loop_extent") {
    yakl::LoopSpec loop(-1);
    (void) loop;
  } else if (scenario == "loop_stride") {
    yakl::LoopSpec loop(0,10,0);
    (void) loop;
  } else if (scenario == "loop_f_extent") {
    yakl::LoopSpec_F loop(-1);
    (void) loop;
  } else if (scenario == "loop_f_stride") {
    yakl::LoopSpec_F loop(-3,7,0);
    (void) loop;
  } else if (scenario == "simple_bounds_negative") {
    yakl::SimpleBounds<1> bounds(-1);
    (void) bounds;
  } else if (scenario == "simple_bounds_overflow") {
    yakl::SimpleBounds<2> bounds(std::numeric_limits<size_t>::max(),2);
    (void) bounds;
  } else if (scenario == "index_width_overflow") {
    yakl::LoopSpec loop(std::numeric_limits<uint64_t>::max());
    (void) loop;
  } else if (scenario == "linear_allocator") {
    yakl::LinearAllocator allocator(1024,0);
    (void) allocator;
  } else if (scenario == "linear_allocator_zero_pool") {
    yakl::LinearAllocator allocator(0,yakl::LinearAllocator::requiredAlignment);
    (void) allocator;
  } else if (scenario == "linear_allocator_small_block") {
    yakl::LinearAllocator allocator(1024,2*sizeof(size_t));
    (void) allocator;
  } else if (scenario == "linear_allocator_overflow") {
    yakl::LinearAllocator allocator(std::numeric_limits<size_t>::max(),yakl::LinearAllocator::requiredAlignment);
    (void) allocator;
  } else if (scenario == "linear_allocator_allocation_overflow") {
    yakl::LinearAllocator allocator(1024,yakl::LinearAllocator::requiredAlignment);
    (void) allocator.allocate(std::numeric_limits<size_t>::max());
  } else if (scenario == "linear_allocator_exhaustion") {
    yakl::LinearAllocator allocator(256,yakl::LinearAllocator::requiredAlignment);
    (void) allocator.allocate(257);
  } else if (scenario == "linear_allocator_invalid_free") {
    yakl::LinearAllocator allocator(256,yakl::LinearAllocator::requiredAlignment);
    auto *ptr = static_cast<char *>(allocator.allocate(1));
    (void) allocator.free(ptr+1);
  } else if (scenario == "linear_allocator_double_free") {
    yakl::LinearAllocator allocator(256,yakl::LinearAllocator::requiredAlignment);
    void *ptr = allocator.allocate(1);
    allocator.free(ptr);
    (void) allocator.free(ptr);
  } else if (scenario == "linear_allocator_end_pointer") {
    yakl::LinearAllocator allocator(256,yakl::LinearAllocator::requiredAlignment);
    (void) allocator.getPtr(allocator.nBlocks);
  } else if (scenario == "linear_allocator_uninitialized") {
    yakl::LinearAllocator allocator;
    (void) allocator.allocate(1);
  } else if (scenario == "linear_allocator_empty_callback") {
    yakl::LinearAllocator allocator(256,yakl::LinearAllocator::requiredAlignment,
                                    std::function<void *(size_t)>(),[] (void *) {},[] (void *, size_t) {});
    (void) allocator;
  } else if (scenario == "matinv_singular") {
    yakl::SArray<double,2,2> matrix;
    matrix(0,0) = 1.; matrix(0,1) = 2.;
    matrix(1,0) = 2.; matrix(1,1) = 4.;
    (void) yakl::intrinsics::matinv(matrix);
  } else if (scenario == "matinv_near_singular") {
    yakl::SArray<double,2,2> matrix;
    matrix(0,0) = 1.; matrix(0,1) = 0.;
    matrix(1,0) = 0.; matrix(1,1) = std::numeric_limits<double>::epsilon()/4;
    (void) yakl::intrinsics::matinv(matrix);
  } else if (scenario == "minloc_nan_sarray") {
    yakl::SArray<double,3> values;
    values(0) = 2.; values(1) = std::numeric_limits<double>::quiet_NaN(); values(2) = -1.;
    (void) yakl::intrinsics::minloc(values);
  } else if (scenario == "maxloc_nan_sarray") {
    SArray_F<double,Bnds{-2,0}> values;
    values(-2) = 2.; values(-1) = std::numeric_limits<double>::quiet_NaN(); values(0) = -1.;
    (void) yakl::intrinsics::maxloc(values);
  } else if (scenario == "minloc_nan_array") {
    Array<double *,Kokkos::HostSpace> host("minloc NaN",3);
    host(0) = 2.; host(1) = std::numeric_limits<double>::quiet_NaN(); host(2) = -1.;
    auto values = host.createDeviceCopy();
    (void) yakl::intrinsics::minloc(values);
  } else if (scenario == "maxloc_nan_array") {
    Array_F<double *,Kokkos::HostSpace> host("maxloc NaN",{-2,0});
    host(-2) = 2.; host(-1) = std::numeric_limits<double>::quiet_NaN(); host(0) = -1.;
    auto values = host.createDeviceCopy();
    (void) yakl::intrinsics::maxloc(values);
  } else if (scenario == "random_range") {
    yakl::Random random(1368976481,0);
    (void) random.gen_uniform<double>(2.,1.);
  } else if (scenario == "random_uniform_nan_lower") {
    yakl::Random random(1368976481,0);
    (void) random.gen_uniform<double>(std::numeric_limits<double>::quiet_NaN(),1.);
  } else if (scenario == "random_uniform_nan_upper") {
    yakl::Random random(1368976481,0);
    (void) random.gen_uniform<double>(0.,std::numeric_limits<double>::quiet_NaN());
  } else if (scenario == "random_normal_stddev") {
    yakl::Random random(1368976481,0);
    (void) random.gen_normal<double>(0.,-1.);
  } else if (scenario == "random_normal_nan_mean") {
    yakl::Random random(1368976481,0);
    (void) random.gen_normal<double>(std::numeric_limits<double>::quiet_NaN(),1.);
  } else if (scenario == "random_normal_nan_stddev") {
    yakl::Random random(1368976481,0);
    (void) random.gen_normal<double>(0.,std::numeric_limits<double>::quiet_NaN());
  } else if (scenario == "random_bernoulli_probability") {
    yakl::Random random(1368976481,0);
    (void) random.gen_bernoulli(1.01);
  } else if (scenario == "random_bernoulli_nan") {
    yakl::Random random(1368976481,0);
    (void) random.gen_bernoulli(std::numeric_limits<double>::quiet_NaN());
  } else if (scenario == "random_exponential_rate") {
    yakl::Random random(1368976481,0);
    (void) random.gen_exponential<double>(0.);
  } else if (scenario == "random_exponential_nan") {
    yakl::Random random(1368976481,0);
    (void) random.gen_exponential<double>(std::numeric_limits<double>::quiet_NaN());
  } else if (scenario == "random_lognormal_stddev") {
    yakl::Random random(1368976481,0);
    (void) random.gen_lognormal<double>(0.,-1.);
  } else if (scenario == "random_lognormal_nan_stddev") {
    yakl::Random random(1368976481,0);
    (void) random.gen_lognormal<double>(0.,std::numeric_limits<double>::quiet_NaN());
  } else if (scenario == "timer_stop") {
    yakl::Toney timer;
    timer.stop("inactive");
  } else if (scenario == "unpack_index") {
    yakl::SimpleBounds<2> bounds(2,3);
    yakl::uindex_t i, j;
    bounds.unpack(bounds.nIter,i,j);
  } else if (scenario == "autotune_index") {
    (void) yakl::autotune::get_config(-1);
  } else if (scenario == "config_tile_zero") {
    (void) yakl::Config<128>(0);
  } else if (scenario == "config_tile_negative") {
    (void) yakl::Config<128>(-1);
  } else if (scenario == "finalize_with_live_allocation") {
    Array<int *,yakl::DeviceSpace> arr("live allocation",1);
    yakl::finalize();
  } else {
    return 2;
  }

  yakl::finalize();
  Kokkos::finalize();
  #ifdef HAVE_MPI
    MPI_Finalize();
  #endif
  return 0;
}
