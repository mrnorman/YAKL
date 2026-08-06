
#include <iostream>
#include "YAKL.h"
#include <random>
#include <algorithm>

using yakl::Array_F;
using yakl::COLON;

static_assert(std::same_as<std::remove_cv_t<decltype(COLON)>,Kokkos::ALL_t>);
using yakl::parallel_for_F;
using yakl::Bounds_F;
using yakl::SimpleBounds_F;
using yakl::Bnds;

typedef float real;

typedef Array_F<real *       ,Kokkos::HostSpace> realHost1d;
typedef Array_F<real **      ,Kokkos::HostSpace> realHost2d;
typedef Array_F<real ***     ,Kokkos::HostSpace> realHost3d;
typedef Array_F<real ****    ,Kokkos::HostSpace> realHost4d;
typedef Array_F<real *****   ,Kokkos::HostSpace> realHost5d;
typedef Array_F<real ******  ,Kokkos::HostSpace> realHost6d;
typedef Array_F<real ******* ,Kokkos::HostSpace> realHost7d;
typedef Array_F<real ********,Kokkos::HostSpace> realHost8d;

typedef Array_F<real *       ,yakl::DeviceSpace> real1d;
typedef Array_F<real **      ,yakl::DeviceSpace> real2d;
typedef Array_F<real ***     ,yakl::DeviceSpace> real3d;
typedef Array_F<real ****    ,yakl::DeviceSpace> real4d;
typedef Array_F<real *****   ,yakl::DeviceSpace> real5d;
typedef Array_F<real ******  ,yakl::DeviceSpace> real6d;
typedef Array_F<real ******* ,yakl::DeviceSpace> real7d;
typedef Array_F<real ********,yakl::DeviceSpace> real8d;

void die(std::string msg) {
  Kokkos::abort(msg.c_str());
}


auto construct_const_array_host() {
  return Array_F<real const *,Kokkos::HostSpace>( realHost1d("arr",10) );
}


auto construct_const_array_device() {
  return Array_F<real const *,yakl::DeviceSpace>( real1d("arr",10) );
}


int main() {
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");
    int constexpr d8 = 2;
    int constexpr d7 = 3;
    int constexpr d6 = 4;
    int constexpr d5 = 5;
    int constexpr d4 = 6;
    int constexpr d3 = 7;
    int constexpr d2 = 8;
    int constexpr d1 = 9;

    ///////////////////////////////////////////////////////////
    // Test operator()
    ///////////////////////////////////////////////////////////

    real1d test1d("test1d",d1);
    real2d test2d("test2d",d1,d2);
    real3d test3d("test3d",d1,d2,d3);
    real4d test4d("test4d",d1,d2,d3,d4);
    real5d test5d("test5d",d1,d2,d3,d4,d5);
    real6d test6d("test6d",d1,d2,d3,d4,d5,d6);
    real7d test7d("test7d",d1,d2,d3,d4,d5,d6,d7);
    real8d test8d("test8d",d1,d2,d3,d4,d5,d6,d7,d8);

    std::cout << "Is SArray_F trivially copyable? " << std::is_trivially_copyable<yakl::SArray_F<real,Bnds{1,1}>>::value << std::endl;

    test1d = 0.f;
    test2d = 0.f;
    test3d = 0.f;
    test4d = 0.f;
    test5d = 0.f;
    test6d = 0.f;
    test7d = 0.f;
    test8d = 0.f;


    parallel_for_F( Bounds_F<1>(d1) , KOKKOS_LAMBDA (int i1) {
      test1d(i1) = 1;
    });
    parallel_for_F( Bounds_F<2>(d1,d2) , KOKKOS_LAMBDA (int i1, int i2) {
      test2d(i1,i2) = 1;
    });
    parallel_for_F( Bounds_F<3>(d1,d2,d3) , KOKKOS_LAMBDA (int i1, int i2, int i3) {
      test3d(i1,i2,i3) = 1;
    });
    parallel_for_F( Bounds_F<4>(d1,d2,d3,d4) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4) {
      test4d(i1,i2,i3,i4) = 1;
    });
    parallel_for_F( Bounds_F<5>(d1,d2,d3,d4,d5) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5) {
      test5d(i1,i2,i3,i4,i5) = 1;
    });
    parallel_for_F( Bounds_F<6>(d1,d2,d3,d4,d5,d6) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6) {
      test6d(i1,i2,i3,i4,i5,i6) = 1;
    });
    parallel_for_F( Bounds_F<7>(d1,d2,d3,d4,d5,d6,d7) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6, int i7) {
      test7d(i1,i2,i3,i4,i5,i6,i7) = 1;
    });
    parallel_for_F( Bounds_F<8>(d1,d2,d3,d4,d5,d6,d7,d8) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8) {
      test8d(i1,i2,i3,i4,i5,i6,i7,i8) = 1;
    });

    if (yakl::intrinsics::sum(test1d) != d1                     ) { die("LOOPS: wrong sum for test1d"); }
    if (yakl::intrinsics::sum(test2d) != d1*d2                  ) { die("LOOPS: wrong sum for test2d"); }
    if (yakl::intrinsics::sum(test3d) != d1*d2*d3               ) { die("LOOPS: wrong sum for test3d"); }
    if (yakl::intrinsics::sum(test4d) != d1*d2*d3*d4            ) { die("LOOPS: wrong sum for test4d"); }
    if (yakl::intrinsics::sum(test5d) != d1*d2*d3*d4*d5         ) { die("LOOPS: wrong sum for test5d"); }
    if (yakl::intrinsics::sum(test6d) != d1*d2*d3*d4*d5*d6      ) { die("LOOPS: wrong sum for test6d"); }
    if (yakl::intrinsics::sum(test7d) != d1*d2*d3*d4*d5*d6*d7   ) { die("LOOPS: wrong sum for test7d"); }
    if (yakl::intrinsics::sum(test8d) != d1*d2*d3*d4*d5*d6*d7*d8) { die("LOOPS: wrong sum for test8d"); }

    if (test1d.rank() != 1) { die("Ranks: wrong rank for test1d"); }
    if (test2d.rank() != 2) { die("Ranks: wrong rank for test2d"); }
    if (test3d.rank() != 3) { die("Ranks: wrong rank for test3d"); }
    if (test4d.rank() != 4) { die("Ranks: wrong rank for test4d"); }
    if (test5d.rank() != 5) { die("Ranks: wrong rank for test5d"); }
    if (test6d.rank() != 6) { die("Ranks: wrong rank for test6d"); }
    if (test7d.rank() != 7) { die("Ranks: wrong rank for test7d"); }
    if (test8d.rank() != 8) { die("Ranks: wrong rank for test8d"); }

    if (test1d.size() != d1                     ) { die("get_elem_count: wrong value for test1d"); }
    if (test2d.size() != d1*d2                  ) { die("get_elem_count: wrong value for test2d"); }
    if (test3d.size() != d1*d2*d3               ) { die("get_elem_count: wrong value for test3d"); }
    if (test4d.size() != d1*d2*d3*d4            ) { die("get_elem_count: wrong value for test4d"); }
    if (test5d.size() != d1*d2*d3*d4*d5         ) { die("get_elem_count: wrong value for test5d"); }
    if (test6d.size() != d1*d2*d3*d4*d5*d6      ) { die("get_elem_count: wrong value for test6d"); }
    if (test7d.size() != d1*d2*d3*d4*d5*d6*d7   ) { die("get_elem_count: wrong value for test7d"); }
    if (test8d.size() != d1*d2*d3*d4*d5*d6*d7*d8) { die("get_elem_count: wrong value for test8d"); }

    if (yakl::intrinsics::sum(test1d.extents()) != d1                     ) { die("get_dimensions: wrong value for test1d"); }
    if (yakl::intrinsics::sum(test2d.extents()) != d1+d2                  ) { die("get_dimensions: wrong value for test2d"); }
    if (yakl::intrinsics::sum(test3d.extents()) != d1+d2+d3               ) { die("get_dimensions: wrong value for test3d"); }
    if (yakl::intrinsics::sum(test4d.extents()) != d1+d2+d3+d4            ) { die("get_dimensions: wrong value for test4d"); }
    if (yakl::intrinsics::sum(test5d.extents()) != d1+d2+d3+d4+d5         ) { die("get_dimensions: wrong value for test5d"); }
    if (yakl::intrinsics::sum(test6d.extents()) != d1+d2+d3+d4+d5+d6      ) { die("get_dimensions: wrong value for test6d"); }
    if (yakl::intrinsics::sum(test7d.extents()) != d1+d2+d3+d4+d5+d6+d7   ) { die("get_dimensions: wrong value for test7d"); }
    if (yakl::intrinsics::sum(test8d.extents()) != d1+d2+d3+d4+d5+d6+d7+d8) { die("get_dimensions: wrong value for test8d"); }

    if (test1d.extent(0) != d1) { die("extent: wrong value for test1d"); }
    if (test2d.extent(1) != d2) { die("extent: wrong value for test2d"); }
    if (test3d.extent(2) != d3) { die("extent: wrong value for test3d"); }
    if (test4d.extent(3) != d4) { die("extent: wrong value for test4d"); }
    if (test5d.extent(4) != d5) { die("extent: wrong value for test5d"); }
    if (test6d.extent(5) != d6) { die("extent: wrong value for test6d"); }
    if (test7d.extent(6) != d7) { die("extent: wrong value for test7d"); }
    if (test8d.extent(7) != d8) { die("extent: wrong value for test8d"); }

    ///////////////////////////////////////////////////////////
    // Test host-side reference counting for transformed Arrays
    ///////////////////////////////////////////////////////////
    {
      real2d retained;
      {
        real1d source("reshape owner",{-5,6});
        source = 3;
        if (source.use_count() != 1) die("reshape: source should begin with reference count 1");
        retained = source.reshape({2,4},{-3,0});
        if (source.use_count() != 2 || retained.use_count() != 2) {
          die("reshape: transformed Array_F did not retain its source allocation");
        }
        if (retained.label() != source.label()) die("reshape: transformed Array_F did not retain its label");
        if (retained.lbounds()(1) != 2 || retained.lbounds()(2) != -3) {
          die("reshape: transformed Array_F did not retain its requested lower bounds");
        }
        auto copy = retained;
        if (source.use_count() != 3 || retained.use_count() != 3 || copy.use_count() != 3) {
          die("reshape: copying transformed Array_F produced an incorrect reference count");
        }
      }
      if (retained.use_count() != 1) die("reshape: retained Array_F should be sole owner after source destruction");
      if (yakl::intrinsics::sum(retained) != 36) die("reshape: retained Array_F became invalid after source destruction");
    }
    {
      real1d retained;
      {
        real2d source("collapse owner",{-1,1},{4,7});
        source = 4;
        retained = source.collapse(-6);
        if (source.use_count() != 2 || retained.use_count() != 2) {
          die("collapse: transformed Array_F did not retain its source allocation");
        }
        if (retained.data() != source.data() || retained.lbounds()(1) != -6) {
          die("collapse: transformed Array_F has incorrect pointer or lower bound");
        }
        {
          auto flattened = source.flatten(-6);
          auto default_lb = source.flatten();
          if (source.use_count() != 4 || flattened.use_count() != 4 || default_lb.use_count() != 4) {
            die("flatten: transformed Array_F has an incorrect reference count");
          }
          if (flattened.data() != retained.data() || flattened.lbounds()(1) != -6 || default_lb.lbounds()(1) != 1) {
            die("flatten: result differs from collapse");
          }
        }
        if (source.use_count() != 2) die("flatten: temporary Array_F aliases did not release their references");
      }
      if (retained.use_count() != 1) die("collapse: retained Array_F should be sole owner after source destruction");
      if (yakl::intrinsics::sum(retained) != 48) die("collapse: retained Array_F became invalid after source destruction");
    }
    {
      real1d retained;
      {
        real2d source("slice owner",{-1,1},{4,7});
        source = 5;
        retained = source.slice<1>(COLON,5);
        if (source.use_count() != 2 || retained.use_count() != 2) {
          die("slice: transformed Array_F did not retain its source allocation");
        }
        if (retained.data() != source.data()+source.stride(1) || retained.lbounds()(1) != -1) {
          die("slice: transformed Array_F has incorrect offset or lower bound");
        }
        {
          auto ignored_index = source.slice<1>(-987654,5);
          if (source.use_count() != 3 || ignored_index.use_count() != 3) {
            die("slice: numeric index ignored for a whole Array_F dimension has an incorrect reference count");
          }
          if (ignored_index.data() != retained.data() || ignored_index.lbounds()(1) != retained.lbounds()(1)) {
            die("slice: numeric index supplied for a whole Array_F dimension was not ignored");
          }
        }
        if (source.use_count() != 2) die("slice: temporary Array_F alias did not release its reference");
      }
      if (retained.use_count() != 1) die("slice: retained Array_F should be sole owner after source destruction");
      if (yakl::intrinsics::sum(retained) != 15) die("slice: retained Array_F became invalid after source destruction");
    }
    {
      real2d retained;
      {
        real2d source("subset owner",{-1,1},{4,7});
        source = 6;
        retained = source.subset_slowest_dimension(5,6);
        if (source.use_count() != 2 || retained.use_count() != 2) {
          die("subset: transformed Array_F did not retain its source allocation");
        }
        if (retained.data() != source.data()+source.stride(1) || retained.lbounds()(2) != 5) {
          die("subset: transformed Array_F has incorrect offset or lower bound");
        }
      }
      if (retained.use_count() != 1) die("subset: retained Array_F should be sole owner after source destruction");
      if (yakl::intrinsics::sum(retained) != 36) die("subset: retained Array_F became invalid after source destruction");
    }
    {
      real2d source("device transformations",{-1,0},{4,6});
      Array_F<int *,yakl::DeviceSpace> result("device transformation result",1);
      source = 2;
      if (source.use_count() != 1) die("device transformations: source should begin with reference count 1");
      parallel_for_F( SimpleBounds_F<1>(1) , KOKKOS_LAMBDA (int i) {
        auto reshaped  = source.reshape({10,12},{-4,-3});
        auto collapsed = source.collapse(-2);
        auto sliced    = source.slice<1>(COLON,5);
        auto subset    = source.subset_slowest_dimension(5,5);
        result(i) = reshaped(12,-3) + collapsed(3) + sliced(-1) + subset(-1,5);
      });
      Kokkos::fence();
      if (source.use_count() != 1) die("device transformations: device-local aliases changed the host reference count");
      if (result.createHostCopy()(1) != 8) die("device transformations: a device-local transformed Array_F is invalid");
    }

    ///////////////////////////////////////////////////////////
    // Test unmanaged arrays
    ///////////////////////////////////////////////////////////
    real1d test1d_ptr(test1d.data(),d1);
    real2d test2d_ptr(test2d.data(),d1,d2);
    real3d test3d_ptr(test3d.data(),d1,d2,d3);
    real4d test4d_ptr(test4d.data(),d1,d2,d3,d4);
    real5d test5d_ptr(test5d.data(),d1,d2,d3,d4,d5);
    real6d test6d_ptr(test6d.data(),d1,d2,d3,d4,d5,d6);
    real7d test7d_ptr(test7d.data(),d1,d2,d3,d4,d5,d6,d7);
    real8d test8d_ptr(test8d.data(),d1,d2,d3,d4,d5,d6,d7,d8);

    auto unmanaged_reshape  = test2d_ptr.reshape(d2,d1);
    auto unmanaged_collapse = test2d_ptr.collapse(1);
    auto unmanaged_slice    = test2d_ptr.slice<1>(COLON,2);
    auto unmanaged_subset   = test2d_ptr.subset_slowest_dimension(2,2);
    if (test2d_ptr.use_count() != 0 || unmanaged_reshape.use_count() != 0 || unmanaged_collapse.use_count() != 0 ||
        unmanaged_slice.use_count() != 0 || unmanaged_subset.use_count() != 0) {
      die("UNMANAGED: transforming an unmanaged Array_F must not manufacture ownership");
    }

    test1d_ptr = 0.f;
    test2d_ptr = 0.f;
    test3d_ptr = 0.f;
    test4d_ptr = 0.f;
    test5d_ptr = 0.f;
    test6d_ptr = 0.f;
    test7d_ptr = 0.f;
    test8d_ptr = 0.f;

    parallel_for_F( Bounds_F<1>(d1) , KOKKOS_LAMBDA (int i1) {
      test1d_ptr(i1) = 1;
    });
    parallel_for_F( Bounds_F<2>(d1,d2) , KOKKOS_LAMBDA (int i1, int i2) {
      test2d_ptr(i1,i2) = 1;
    });
    parallel_for_F( Bounds_F<3>(d1,d2,d3) , KOKKOS_LAMBDA (int i1, int i2, int i3) {
      test3d_ptr(i1,i2,i3) = 1;
    });
    parallel_for_F( Bounds_F<4>(d1,d2,d3,d4) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4) {
      test4d_ptr(i1,i2,i3,i4) = 1;
    });
    parallel_for_F( Bounds_F<5>(d1,d2,d3,d4,d5) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5) {
      test5d_ptr(i1,i2,i3,i4,i5) = 1;
    });
    parallel_for_F( Bounds_F<6>(d1,d2,d3,d4,d5,d6) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6) {
      test6d_ptr(i1,i2,i3,i4,i5,i6) = 1;
    });
    parallel_for_F( Bounds_F<7>(d1,d2,d3,d4,d5,d6,d7) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6, int i7) {
      test7d_ptr(i1,i2,i3,i4,i5,i6,i7) = 1;
    });
    parallel_for_F( Bounds_F<8>(d1,d2,d3,d4,d5,d6,d7,d8) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8) {
      test8d_ptr(i1,i2,i3,i4,i5,i6,i7,i8) = 1;
    });

    if (yakl::intrinsics::sum(test1d) != d1                     ) { die("UNMANAGED: wrong sum for test1d"); }
    if (yakl::intrinsics::sum(test2d) != d1*d2                  ) { die("UNMANAGED: wrong sum for test2d"); }
    if (yakl::intrinsics::sum(test3d) != d1*d2*d3               ) { die("UNMANAGED: wrong sum for test3d"); }
    if (yakl::intrinsics::sum(test4d) != d1*d2*d3*d4            ) { die("UNMANAGED: wrong sum for test4d"); }
    if (yakl::intrinsics::sum(test5d) != d1*d2*d3*d4*d5         ) { die("UNMANAGED: wrong sum for test5d"); }
    if (yakl::intrinsics::sum(test6d) != d1*d2*d3*d4*d5*d6      ) { die("UNMANAGED: wrong sum for test6d"); }
    if (yakl::intrinsics::sum(test7d) != d1*d2*d3*d4*d5*d6*d7   ) { die("UNMANAGED: wrong sum for test7d"); }
    if (yakl::intrinsics::sum(test8d) != d1*d2*d3*d4*d5*d6*d7*d8) { die("UNMANAGED: wrong sum for test8d"); }

    ///////////////////////////////////////////////////////////
    // Test createHostCopy();
    ///////////////////////////////////////////////////////////
    auto testHost1d = test1d.createHostCopy();
    auto testHost2d = test2d.createHostCopy();
    auto testHost3d = test3d.createHostCopy();
    auto testHost4d = test4d.createHostCopy();
    auto testHost5d = test5d.createHostCopy();
    auto testHost6d = test6d.createHostCopy();
    auto testHost7d = test7d.createHostCopy();
    auto testHost8d = test8d.createHostCopy();

    if (yakl::intrinsics::sum(testHost1d) != d1                     ) { die("createHostCopy: wrong sum for testHost1d"); }
    if (yakl::intrinsics::sum(testHost2d) != d1*d2                  ) { die("createHostCopy: wrong sum for testHost2d"); }
    if (yakl::intrinsics::sum(testHost3d) != d1*d2*d3               ) { die("createHostCopy: wrong sum for testHost3d"); }
    if (yakl::intrinsics::sum(testHost4d) != d1*d2*d3*d4            ) { die("createHostCopy: wrong sum for testHost4d"); }
    if (yakl::intrinsics::sum(testHost5d) != d1*d2*d3*d4*d5         ) { die("createHostCopy: wrong sum for testHost5d"); }
    if (yakl::intrinsics::sum(testHost6d) != d1*d2*d3*d4*d5*d6      ) { die("createHostCopy: wrong sum for testHost6d"); }
    if (yakl::intrinsics::sum(testHost7d) != d1*d2*d3*d4*d5*d6*d7   ) { die("createHostCopy: wrong sum for testHost7d"); }
    if (yakl::intrinsics::sum(testHost8d) != d1*d2*d3*d4*d5*d6*d7*d8) { die("createHostCopy: wrong sum for testHost8d"); }

    ///////////////////////////////////////////////////////////
    // Test host memset
    ///////////////////////////////////////////////////////////
    testHost8d = 0.f;
    if (yakl::intrinsics::sum(testHost8d) != 0) { die("memset: failed for testHost8d"); }

    ///////////////////////////////////////////////////////////
    // Test deep_copy_to device to host
    ///////////////////////////////////////////////////////////
    test8d.deep_copy_to(testHost8d);
    Kokkos::fence();
    if (yakl::intrinsics::sum(testHost8d) != d1*d2*d3*d4*d5*d6*d7*d8) { die("deep_copy_to: wrong sum for testHost8d"); }

    ///////////////////////////////////////////////////////////
    // Test device memset
    ///////////////////////////////////////////////////////////
    test8d = 0.f;
    if (yakl::intrinsics::sum(test8d) != 0) { die("memset: failed for test8d"); }

    ///////////////////////////////////////////////////////////
    // Test deep_copy_to host to device
    ///////////////////////////////////////////////////////////
    testHost8d.deep_copy_to(test8d);
    Kokkos::fence();
    if (yakl::intrinsics::sum(test8d) != d1*d2*d3*d4*d5*d6*d7*d8) { die("deep_copy_to: wrong sum for test8d"); }

    ///////////////////////////////////////////////////////////
    // Test createDeviceCopy from device
    ///////////////////////////////////////////////////////////
    auto test8d_dev2 = test8d.createDeviceCopy();
    if (yakl::intrinsics::sum(test8d_dev2) != d1*d2*d3*d4*d5*d6*d7*d8) { die("createDeviceCopy: wrong sum for test8d_dev2"); }

    ///////////////////////////////////////////////////////////
    // Test deep_copy_to device to device
    ///////////////////////////////////////////////////////////
    test8d_dev2 = 0.f;
    test8d.deep_copy_to(test8d_dev2);
    Kokkos::fence();
    if (yakl::intrinsics::sum(test8d_dev2) != d1*d2*d3*d4*d5*d6*d7*d8) { die("deep_copy_to: wrong sum for test8d_dev2"); }

    ///////////////////////////////////////////////////////////
    // Test slice
    ///////////////////////////////////////////////////////////
    test8d = 0.f;
    auto slice = test8d.slice<3>(COLON,COLON,COLON,6,5,4,3,2);
    slice = 1.f;
    if (yakl::intrinsics::sum(test8d) != d1*d2*d3) { die("slice: wrong sum for slice"); }

    ///////////////////////////////////////////////////////////
    // Test slice inside a kernel
    ///////////////////////////////////////////////////////////
    test8d = 0.f;
    parallel_for_F( 1 , KOKKOS_LAMBDA (int dummy) {
      auto slice = test8d.slice<3>(COLON,COLON,COLON,6,5,4,3,2);
    });

    ///////////////////////////////////////////////////////////
    // Test non-1 lower bounds and non-standard loops
    ///////////////////////////////////////////////////////////
    real3d lower("lower",{-1,2},5,{0,4} );
    lower = 0.f;
    parallel_for_F( Bounds_F<3>({-1,2,2},5,{0,4}) , KOKKOS_LAMBDA (int i, int j, int k) {
      lower(i+1,j,k) = 1;
    });
    if (yakl::intrinsics::sum(lower) != 50) { die("lower bounds: incorrect sum for lower"); }

    ///////////////////////////////////////////////////////////
    // get_lbounds and get_ubounds
    ///////////////////////////////////////////////////////////
    auto lbnds = lower.lbounds();
    auto ubnds = lower.ubounds();
    if (lbnds(1) != -1 || lbnds(2) != 1 || lbnds(3) != 0) { die("get_lbounds: wrong lower bounds for lower"); }
    if (ubnds(1) != 2  || ubnds(2) != 5 || ubnds(3) != 4) { die("get_ubounds: wrong upper bounds for lower"); }

    ///////////////////////////////////////////////////////////
    // Test SimpleBounds_F
    ///////////////////////////////////////////////////////////
    test1d = 0.f;
    test2d = 0.f;
    test3d = 0.f;
    test4d = 0.f;
    test5d = 0.f;
    test6d = 0.f;
    test7d = 0.f;
    test8d = 0.f;

    parallel_for_F( SimpleBounds_F<1>(d1) , KOKKOS_LAMBDA (int i1) {
      test1d(i1) = 1;
    });
    parallel_for_F( SimpleBounds_F<2>(d1,d2) , KOKKOS_LAMBDA (int i1, int i2) {
      test2d(i1,i2) = 1;
    });
    parallel_for_F( SimpleBounds_F<3>(d1,d2,d3) , KOKKOS_LAMBDA (int i1, int i2, int i3) {
      test3d(i1,i2,i3) = 1;
    });
    parallel_for_F( SimpleBounds_F<4>(d1,d2,d3,d4) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4) {
      test4d(i1,i2,i3,i4) = 1;
    });
    parallel_for_F( SimpleBounds_F<5>(d1,d2,d3,d4,d5) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5) {
      test5d(i1,i2,i3,i4,i5) = 1;
    });
    parallel_for_F( SimpleBounds_F<6>(d1,d2,d3,d4,d5,d6) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6) {
      test6d(i1,i2,i3,i4,i5,i6) = 1;
    });
    parallel_for_F( SimpleBounds_F<7>(d1,d2,d3,d4,d5,d6,d7) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6, int i7) {
      test7d(i1,i2,i3,i4,i5,i6,i7) = 1;
    });
    parallel_for_F( SimpleBounds_F<8>(d1,d2,d3,d4,d5,d6,d7,d8) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8) {
      test8d(i1,i2,i3,i4,i5,i6,i7,i8) = 1;
    });

    if (yakl::intrinsics::sum(test1d) != d1                     ) { die("LOOPS: wrong sum for test1d"); }
    if (yakl::intrinsics::sum(test2d) != d1*d2                  ) { die("LOOPS: wrong sum for test2d"); }
    if (yakl::intrinsics::sum(test3d) != d1*d2*d3               ) { die("LOOPS: wrong sum for test3d"); }
    if (yakl::intrinsics::sum(test4d) != d1*d2*d3*d4            ) { die("LOOPS: wrong sum for test4d"); }
    if (yakl::intrinsics::sum(test5d) != d1*d2*d3*d4*d5         ) { die("LOOPS: wrong sum for test5d"); }
    if (yakl::intrinsics::sum(test6d) != d1*d2*d3*d4*d5*d6      ) { die("LOOPS: wrong sum for test6d"); }
    if (yakl::intrinsics::sum(test7d) != d1*d2*d3*d4*d5*d6*d7   ) { die("LOOPS: wrong sum for test7d"); }
    if (yakl::intrinsics::sum(test8d) != d1*d2*d3*d4*d5*d6*d7*d8) { die("LOOPS: wrong sum for test8d"); }

    // if (wrap.get_lbounds()(1) != 2) die("wrap wrong lbounds");


    ///////////////////////////////////////////////////////////
    // Test reshape
    ///////////////////////////////////////////////////////////
    auto reshaped = test8d.reshape({2,21},16,{-1,1132});
    reshaped = 2.f;
    if (yakl::intrinsics::sum(test8d) != d1*d2*d3*d4*d5*d6*d7*d8*2) { die("SimpleBounds_F: wrong sum for reshaped test8d"); }


    ///////////////////////////////////////////////////////////
    // Test collapse
    ///////////////////////////////////////////////////////////
    auto collapsed = test8d.collapse(-1);
    collapsed = 3.f;
    if (yakl::intrinsics::sum(test8d) != d1*d2*d3*d4*d5*d6*d7*d8*3) { die("SimpleBounds_F: wrong sum for collapsed test8d"); }

    auto constHostArr = construct_const_array_host();
    constHostArr = decltype(constHostArr)();
    if (constHostArr.is_allocated()) die("constHostArr: array didn't deallocate properly");

    auto constDevArr = construct_const_array_device();
    constDevArr = decltype(constDevArr)();
    if (constDevArr.is_allocated()) die("constDevArr: array didn't deallocate properly");

    ///////////////////////////////////////////////////////////
    // Test subset_slowest_dimension
    ///////////////////////////////////////////////////////////
    test7d = 1;
    test7d.subset_slowest_dimension(2) = 2;
    if (yakl::intrinsics::sum(test7d) != d1*d2*d3*d4*d5*d6*5) { die("SimpleBounds_F: wrong sum for reshaped subset"); }

    {
      yakl::Array_F<int *,Kokkos::HostSpace> indices("indices",100);
      for (int i=1; i <= 100; i++) { indices(i) = i; }
      std::shuffle( indices.begin() , indices.end() , std::default_random_engine(13) );
      int tot = 0;
      for (int i=1; i <= 99; i++) { tot += std::abs( indices(i+1) - indices(i) ); }
      if (tot == 99) die("ERROR: Shuffle did not work for FArray");
    }

    {
      yakl::SArray_F<int,Bnds{1,100}> indices;
      for (int i=1; i <= 100; i++) { indices(i) = i; }
      std::shuffle( indices.begin() , indices.end() , std::default_random_engine(13) );
      int tot = 0;
      for (int i=1; i <= 99; i++) { tot += std::abs( indices(i+1) - indices(i) ); }
      if (tot == 99) die("ERROR: Shuffle did not work for FArray");
    }

    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize(); 
  
  return 0;
}
