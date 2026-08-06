#include <iostream>
#include <string>
#include "YAKL.h"

using yakl::Array;
using yakl::Array_F;
using yakl::Bnds;
using yakl::SArray_F;

void fail(std::string const &message) {
  Kokkos::abort(message.c_str());
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  std::string const scenario = argv[1];
  Kokkos::initialize();

  if (scenario == "allocate_before_init") {
    (void) yakl::alloc_device(16,"outside YAKL lifetime");
    return 0;
  } else if (scenario == "free_before_init") {
    void * ptr = Kokkos::kokkos_malloc("outside YAKL lifetime",16);
    yakl::free_device(ptr,"outside YAKL lifetime");
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

    SArray_F<int,Bnds{-10000000000LL,-9999999998LL},Bnds{-7,-4}> stack;
    for (size_t linear=0; linear < stack.size(); linear++) {
      stack.data()[linear] = static_cast<int>(linear);
      auto index = stack.unpack_global_index(linear);
      if (stack(index(1),index(2)) != stack.data()[linear]) {
        fail("SArray_F large negative lower-bound validation failed");
      }
    }

    yakl::Bounds_F<2> bounds({-7,5,3},{11,20,4});
    if (bounds.nIter != 8) fail("strided Bounds iteration count is incorrect");
    for (size_t linear=0; linear < bounds.nIter; linear++) {
      ptrdiff_t i, j;
      bounds.unpack(linear,i,j);
      if (i < -7 || i > 5 || (i+7)%3 != 0 || j < 11 || j > 20 || (j-11)%4 != 0) {
        fail("strided Bounds unpack returned an invalid index");
      }
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
  } else if (scenario == "component_shape") {
    using namespace yakl::componentwise;
    Array<int **,Kokkos::HostSpace> left("left",2,3);
    Array<int **,Kokkos::HostSpace> right("right",3,2);
    auto result = left + right;
    (void) result;
  } else if (scenario == "loop_extent") {
    yakl::LoopSpec<> loop(-1);
    (void) loop;
  } else if (scenario == "loop_stride") {
    yakl::LoopSpec<> loop(0,10,0);
    (void) loop;
  } else if (scenario == "linear_allocator") {
    yakl::LinearAllocator allocator(1024,0);
    (void) allocator;
  } else if (scenario == "random_range") {
    yakl::Random random;
    (void) random.genFP<double>(2.,1.);
  } else if (scenario == "timer_stop") {
    yakl::Toney timer;
    timer.stop("inactive");
  } else if (scenario == "unpack_index") {
    yakl::SimpleBounds<2> bounds(2,3);
    size_t i, j;
    bounds.unpack(bounds.nIter,i,j);
  } else if (scenario == "autotune_index") {
    (void) yakl::autotune::get_config(-1);
  } else if (scenario == "finalize_with_live_allocation") {
    Array<int *,yakl::DeviceSpace> arr("live allocation",1);
    yakl::finalize();
  } else {
    return 2;
  }

  yakl::finalize();
  Kokkos::finalize();
  return 0;
}
