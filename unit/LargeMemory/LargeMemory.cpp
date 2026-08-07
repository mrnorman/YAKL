#include <cstdint>
#include <iostream>
#include <limits>
#include "YAKL.h"

using yakl::Array;
using yakl::Array_F;
using yakl::Bounds_F;
using yakl::DeviceSpace;

void die(std::string const &message) {
  Kokkos::abort(message.c_str());
}

int main() {
  Kokkos::initialize();
  yakl::init();
  {
    size_t constexpr dim0 = 65537;
    size_t constexpr dim1 = 65536;
    size_t constexpr largeSize = dim0*dim1;
    static_assert(largeSize > std::numeric_limits<unsigned int>::max());
    static_assert(sizeof(size_t) > sizeof(unsigned int));

    using value_t = signed char;
    Array<value_t *,DeviceSpace> array(Kokkos::view_alloc(Kokkos::WithoutInitializing,"large array"),largeSize);
    if (array.size() != largeSize || array.extent(0) != largeSize) {
      die("ERROR: large Array extent was truncated");
    }

    yakl::SimpleBounds<1> bounds(largeSize);
    if (bounds.nIter != largeSize) die("ERROR: large SimpleBounds iteration count was truncated");
    size_t lastIndex;
    bounds.unpack(largeSize-1,lastIndex);
    if (lastIndex != largeSize-1) die("ERROR: large SimpleBounds index unpack was truncated");

    yakl::parallel_for("initialize large Array",bounds,KOKKOS_LAMBDA (size_t i) {
      array(i) = 1;
    });

    ptrdiff_t constexpr flower = -1234;
    ptrdiff_t const fupper = flower + static_cast<ptrdiff_t>(largeSize) - 1;
    Array_F<value_t *,DeviceSpace> farray(array.data(),{flower,fupper});
    Bounds_F<1> fbounds(yakl::LoopSpec_F(flower,fupper));
    if (fbounds.nIter != largeSize) die("ERROR: large Bounds_F iteration count was truncated");
    ptrdiff_t lastFortranIndex;
    fbounds.unpack(largeSize-1,lastFortranIndex);
    if (lastFortranIndex != fupper) die("ERROR: large Bounds_F index unpack was truncated");

    yakl::parallel_for_F("mark large Array_F indices",fbounds,KOKKOS_LAMBDA (ptrdiff_t i) {
      if (i == fupper-1) farray(i) = 0;
      if (i == fupper  ) farray(i) = 2;
    });

    using execution_space = typename decltype(array)::execution_space;
    uint64_t sum = 0;
    Kokkos::parallel_reduce(
        "reduce large Array",
        Kokkos::RangePolicy<execution_space,Kokkos::IndexType<size_t>>(0,largeSize),
        KOKKOS_LAMBDA (size_t i, uint64_t &localSum) { localSum += static_cast<uint64_t>(array(i)); },
        Kokkos::Sum<uint64_t>(sum));
    if (sum != largeSize) die("ERROR: large Kokkos parallel_reduce range or Array indexing was truncated");
    if (yakl::intrinsics::minval(array) != 0 || yakl::intrinsics::maxval(array) != 2) {
      die("ERROR: large Array minval or maxval reduction produced the wrong value");
    }

    {
      using yakl::componentwise::operator>;
      auto positive = array > value_t(0);
      if (positive.size() != largeSize || yakl::intrinsics::count(positive) != largeSize-1) {
        die("ERROR: large scalar componentwise operation or count reduction was truncated");
      }
    }
    {
      using yakl::componentwise::operator!;
      auto zero = !array;
      if (zero.size() != largeSize || yakl::intrinsics::count(zero) != 1) {
        die("ERROR: large unary componentwise operation or count reduction was truncated");
      }
    }
    {
      using yakl::componentwise::operator==;
      auto equal = array == farray;
      if (equal.size() != largeSize || yakl::intrinsics::count(equal) != largeSize) {
        die("ERROR: large array componentwise operation or count reduction was truncated");
      }
    }

    auto const cminloc = yakl::intrinsics::minloc(array);
    auto const cmaxloc = yakl::intrinsics::maxloc(array);
    if (cminloc(0) != largeSize-2 || cmaxloc(0) != largeSize-1) {
      die("ERROR: Array minloc or maxloc truncated a large index");
    }

    auto const fminloc = yakl::intrinsics::minloc(farray);
    auto const fmaxloc = yakl::intrinsics::maxloc(farray);
    if (fminloc(1) != fupper-1 || fmaxloc(1) != fupper) {
      die("ERROR: Array_F minloc or maxloc truncated a large Fortran index");
    }
    if (fminloc(1) <= static_cast<ptrdiff_t>(std::numeric_limits<unsigned int>::max())) {
      die("ERROR: Array_F test did not produce an index larger than unsigned int");
    }

    auto const cglobal = array.unpack_global_index(largeSize-1);
    auto const fglobal = farray.unpack_global_index(largeSize-1);
    if (cglobal(0) != largeSize-1 || fglobal(1) != fupper) {
      die("ERROR: unpack_global_index truncated a large index");
    }

    auto array2d = array.reshape(dim0,dim1);
    auto const cglobal2d = array2d.unpack_global_index(largeSize-1);
    if (array2d.size() != largeSize || cglobal2d(0) != dim0-1 || cglobal2d(1) != dim1-1) {
      die("ERROR: large multidimensional Array metadata or unpacking was truncated");
    }

    ptrdiff_t constexpr flower0 = -17;
    ptrdiff_t constexpr flower1 = 23;
    ptrdiff_t const fupper0 = flower0 + static_cast<ptrdiff_t>(dim0) - 1;
    ptrdiff_t const fupper1 = flower1 + static_cast<ptrdiff_t>(dim1) - 1;
    Array_F<value_t **,DeviceSpace> farray2d(array.data(),{flower0,fupper0},{flower1,fupper1});
    auto const fglobal2d = farray2d.unpack_global_index(largeSize-1);
    if (farray2d.size() != largeSize || fglobal2d(1) != fupper0 || fglobal2d(2) != fupper1) {
      die("ERROR: large multidimensional Array_F metadata or unpacking was truncated");
    }

    auto ctail = array.subset_slowest_dimension(largeSize-2,largeSize-1).createHostCopy();
    auto ftail = farray.subset_slowest_dimension(fupper-1,fupper).createHostCopy();
    if (ctail.extent(0) != 2 || ctail(0) != 0 || ctail(1) != 2 ||
        ftail.extent(0) != 2 || ftail(fupper-1) != 0 || ftail(fupper) != 2) {
      die("ERROR: large-index Array or Array_F subset addressed the wrong storage");
    }
  }
  yakl::finalize();
  Kokkos::finalize();
  return 0;
}
