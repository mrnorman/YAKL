
#include <iostream>
#include "YAKL.h"
#include <random>
#include <algorithm>

using yakl::Array;
using yakl::styleFortran;
using yakl::memHost;
using yakl::memDevice;
using yakl::fortran::parallel_for;
using yakl::fortran::Bounds;
using yakl::fortran::SimpleBounds;
using yakl::COLON;

typedef float real;

typedef Array<real,1,memHost,styleFortran> realHost1d;
typedef Array<real,2,memHost,styleFortran> realHost2d;
typedef Array<real,3,memHost,styleFortran> realHost3d;
typedef Array<real,4,memHost,styleFortran> realHost4d;
typedef Array<real,5,memHost,styleFortran> realHost5d;
typedef Array<real,6,memHost,styleFortran> realHost6d;
typedef Array<real,7,memHost,styleFortran> realHost7d;
typedef Array<real,8,memHost,styleFortran> realHost8d;

typedef Array<real,1,memDevice,styleFortran> real1d;
typedef Array<real,2,memDevice,styleFortran> real2d;
typedef Array<real,3,memDevice,styleFortran> real3d;
typedef Array<real,4,memDevice,styleFortran> real4d;
typedef Array<real,5,memDevice,styleFortran> real5d;
typedef Array<real,6,memDevice,styleFortran> real6d;
typedef Array<real,7,memDevice,styleFortran> real7d;
typedef Array<real,8,memDevice,styleFortran> real8d;

void die(std::string msg) {
  Kokkos::abort(msg.c_str());
}


Array<real const,1,memHost,styleFortran> construct_const_array_host() {
  return Array<real const,1,memHost,styleFortran>( realHost1d("arr",10) );
}


Array<real const,1,memDevice,styleFortran> construct_const_array_device() {
  return Array<real const,1,memDevice,styleFortran>( real1d("arr",10) );
}


int main() {
  Kokkos::initialize();
  yakl::init();
  {
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

    std::cout << "Is FSArray trivially copyable? " << std::is_trivially_copyable<yakl::FSArray<real,1,yakl::SB<1>>>::value << std::endl;

    test1d = 0.f;
    test2d = 0.f;
    test3d = 0.f;
    test4d = 0.f;
    test5d = 0.f;
    test6d = 0.f;
    test7d = 0.f;
    test8d = 0.f;


    parallel_for( Bounds<1>(d1) , KOKKOS_LAMBDA (int i1) {
      test1d(i1) = 1;
    });
    parallel_for( Bounds<2>(d1,d2) , KOKKOS_LAMBDA (int i1, int i2) {
      test2d(i1,i2) = 1;
    });
    parallel_for( Bounds<3>(d1,d2,d3) , KOKKOS_LAMBDA (int i1, int i2, int i3) {
      test3d(i1,i2,i3) = 1;
    });
    parallel_for( Bounds<4>(d1,d2,d3,d4) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4) {
      test4d(i1,i2,i3,i4) = 1;
    });
    parallel_for( Bounds<5>(d1,d2,d3,d4,d5) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5) {
      test5d(i1,i2,i3,i4,i5) = 1;
    });
    parallel_for( Bounds<6>(d1,d2,d3,d4,d5,d6) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6) {
      test6d(i1,i2,i3,i4,i5,i6) = 1;
    });
    parallel_for( Bounds<7>(d1,d2,d3,d4,d5,d6,d7) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6, int i7) {
      test7d(i1,i2,i3,i4,i5,i6,i7) = 1;
    });
    parallel_for( Bounds<8>(d1,d2,d3,d4,d5,d6,d7,d8) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8) {
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

    if (test1d.get_rank() != 1) { die("Ranks: wrong rank for test1d"); }
    if (test2d.get_rank() != 2) { die("Ranks: wrong rank for test2d"); }
    if (test3d.get_rank() != 3) { die("Ranks: wrong rank for test3d"); }
    if (test4d.get_rank() != 4) { die("Ranks: wrong rank for test4d"); }
    if (test5d.get_rank() != 5) { die("Ranks: wrong rank for test5d"); }
    if (test6d.get_rank() != 6) { die("Ranks: wrong rank for test6d"); }
    if (test7d.get_rank() != 7) { die("Ranks: wrong rank for test7d"); }
    if (test8d.get_rank() != 8) { die("Ranks: wrong rank for test8d"); }

    if (test1d.get_elem_count() != d1                     ) { die("get_elem_count: wrong value for test1d"); }
    if (test2d.get_elem_count() != d1*d2                  ) { die("get_elem_count: wrong value for test2d"); }
    if (test3d.get_elem_count() != d1*d2*d3               ) { die("get_elem_count: wrong value for test3d"); }
    if (test4d.get_elem_count() != d1*d2*d3*d4            ) { die("get_elem_count: wrong value for test4d"); }
    if (test5d.get_elem_count() != d1*d2*d3*d4*d5         ) { die("get_elem_count: wrong value for test5d"); }
    if (test6d.get_elem_count() != d1*d2*d3*d4*d5*d6      ) { die("get_elem_count: wrong value for test6d"); }
    if (test7d.get_elem_count() != d1*d2*d3*d4*d5*d6*d7   ) { die("get_elem_count: wrong value for test7d"); }
    if (test8d.get_elem_count() != d1*d2*d3*d4*d5*d6*d7*d8) { die("get_elem_count: wrong value for test8d"); }

    if (yakl::intrinsics::sum(test1d.get_dimensions()) != d1                     ) { die("get_dimensions: wrong value for test1d"); }
    if (yakl::intrinsics::sum(test2d.get_dimensions()) != d1+d2                  ) { die("get_dimensions: wrong value for test2d"); }
    if (yakl::intrinsics::sum(test3d.get_dimensions()) != d1+d2+d3               ) { die("get_dimensions: wrong value for test3d"); }
    if (yakl::intrinsics::sum(test4d.get_dimensions()) != d1+d2+d3+d4            ) { die("get_dimensions: wrong value for test4d"); }
    if (yakl::intrinsics::sum(test5d.get_dimensions()) != d1+d2+d3+d4+d5         ) { die("get_dimensions: wrong value for test5d"); }
    if (yakl::intrinsics::sum(test6d.get_dimensions()) != d1+d2+d3+d4+d5+d6      ) { die("get_dimensions: wrong value for test6d"); }
    if (yakl::intrinsics::sum(test7d.get_dimensions()) != d1+d2+d3+d4+d5+d6+d7   ) { die("get_dimensions: wrong value for test7d"); }
    if (yakl::intrinsics::sum(test8d.get_dimensions()) != d1+d2+d3+d4+d5+d6+d7+d8) { die("get_dimensions: wrong value for test8d"); }

    if (test1d.extent(0) != d1) { die("extent: wrong value for test1d"); }
    if (test2d.extent(1) != d2) { die("extent: wrong value for test2d"); }
    if (test3d.extent(2) != d3) { die("extent: wrong value for test3d"); }
    if (test4d.extent(3) != d4) { die("extent: wrong value for test4d"); }
    if (test5d.extent(4) != d5) { die("extent: wrong value for test5d"); }
    if (test6d.extent(5) != d6) { die("extent: wrong value for test6d"); }
    if (test7d.extent(6) != d7) { die("extent: wrong value for test7d"); }
    if (test8d.extent(7) != d8) { die("extent: wrong value for test8d"); }

    ///////////////////////////////////////////////////////////
    // Test unmanaged arrays
    ///////////////////////////////////////////////////////////
    real1d test1d_ptr("test1d",test1d.data(),d1);
    real2d test2d_ptr("test2d",test2d.data(),d1,d2);
    real3d test3d_ptr("test3d",test3d.data(),d1,d2,d3);
    real4d test4d_ptr("test4d",test4d.data(),d1,d2,d3,d4);
    real5d test5d_ptr("test5d",test5d.data(),d1,d2,d3,d4,d5);
    real6d test6d_ptr("test6d",test6d.data(),d1,d2,d3,d4,d5,d6);
    real7d test7d_ptr("test7d",test7d.data(),d1,d2,d3,d4,d5,d6,d7);
    real8d test8d_ptr("test8d",test8d.data(),d1,d2,d3,d4,d5,d6,d7,d8);

    test1d_ptr = 0.f;
    test2d_ptr = 0.f;
    test3d_ptr = 0.f;
    test4d_ptr = 0.f;
    test5d_ptr = 0.f;
    test6d_ptr = 0.f;
    test7d_ptr = 0.f;
    test8d_ptr = 0.f;

    parallel_for( Bounds<1>(d1) , KOKKOS_LAMBDA (int i1) {
      test1d_ptr(i1) = 1;
    });
    parallel_for( Bounds<2>(d1,d2) , KOKKOS_LAMBDA (int i1, int i2) {
      test2d_ptr(i1,i2) = 1;
    });
    parallel_for( Bounds<3>(d1,d2,d3) , KOKKOS_LAMBDA (int i1, int i2, int i3) {
      test3d_ptr(i1,i2,i3) = 1;
    });
    parallel_for( Bounds<4>(d1,d2,d3,d4) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4) {
      test4d_ptr(i1,i2,i3,i4) = 1;
    });
    parallel_for( Bounds<5>(d1,d2,d3,d4,d5) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5) {
      test5d_ptr(i1,i2,i3,i4,i5) = 1;
    });
    parallel_for( Bounds<6>(d1,d2,d3,d4,d5,d6) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6) {
      test6d_ptr(i1,i2,i3,i4,i5,i6) = 1;
    });
    parallel_for( Bounds<7>(d1,d2,d3,d4,d5,d6,d7) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6, int i7) {
      test7d_ptr(i1,i2,i3,i4,i5,i6,i7) = 1;
    });
    parallel_for( Bounds<8>(d1,d2,d3,d4,d5,d6,d7,d8) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8) {
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
    parallel_for( 1 , KOKKOS_LAMBDA (int dummy) {
      auto slice = test8d.slice<3>(COLON,COLON,COLON,6,5,4,3,2);
    });

    ///////////////////////////////////////////////////////////
    // Test non-1 lower bounds and non-standard loops
    ///////////////////////////////////////////////////////////
    real3d lower("lower",{-1,2},5,{0,4} );
    lower = 0.f;
    parallel_for( Bounds<3>({-1,2,2},5,{0,4}) , KOKKOS_LAMBDA (int i, int j, int k) {
      lower(i+1,j,k) = 1;
    });
    if (yakl::intrinsics::sum(lower) != 50) { die("lower bounds: incorrect sum for lower"); }

    ///////////////////////////////////////////////////////////
    // get_lbounds and get_ubounds
    ///////////////////////////////////////////////////////////
    auto lbnds = lower.get_lbounds();
    auto ubnds = lower.get_ubounds();
    if (lbnds(1) != -1 || lbnds(2) != 1 || lbnds(3) != 0) { die("get_lbounds: wrong lower bounds for lower"); }
    if (ubnds(1) != 2  || ubnds(2) != 5 || ubnds(3) != 4) { die("get_ubounds: wrong upper bounds for lower"); }

    ///////////////////////////////////////////////////////////
    // Test SimpleBounds
    ///////////////////////////////////////////////////////////
    test1d = 0.f;
    test2d = 0.f;
    test3d = 0.f;
    test4d = 0.f;
    test5d = 0.f;
    test6d = 0.f;
    test7d = 0.f;
    test8d = 0.f;

    parallel_for( SimpleBounds<1>(d1) , KOKKOS_LAMBDA (int i1) {
      test1d(i1) = 1;
    });
    parallel_for( SimpleBounds<2>(d1,d2) , KOKKOS_LAMBDA (int i1, int i2) {
      test2d(i1,i2) = 1;
    });
    parallel_for( SimpleBounds<3>(d1,d2,d3) , KOKKOS_LAMBDA (int i1, int i2, int i3) {
      test3d(i1,i2,i3) = 1;
    });
    parallel_for( SimpleBounds<4>(d1,d2,d3,d4) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4) {
      test4d(i1,i2,i3,i4) = 1;
    });
    parallel_for( SimpleBounds<5>(d1,d2,d3,d4,d5) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5) {
      test5d(i1,i2,i3,i4,i5) = 1;
    });
    parallel_for( SimpleBounds<6>(d1,d2,d3,d4,d5,d6) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6) {
      test6d(i1,i2,i3,i4,i5,i6) = 1;
    });
    parallel_for( SimpleBounds<7>(d1,d2,d3,d4,d5,d6,d7) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6, int i7) {
      test7d(i1,i2,i3,i4,i5,i6,i7) = 1;
    });
    parallel_for( SimpleBounds<8>(d1,d2,d3,d4,d5,d6,d7,d8) , KOKKOS_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8) {
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

    auto ir_device = test8d.create_ArrayIR();
    auto ir_host   = test8d.createHostCopy().create_ArrayIR();
    if (ir_device.get_memory_type() == array_ir::MEMORY_HOST  ) die("ir_device has wrong memory type");
    if (ir_host  .get_memory_type() == array_ir::MEMORY_DEVICE) die("ir_host   has wrong memory type");
    if (ir_host.extent(1) != test8d.extent(6)) die("ir_host extent(1) is wrong");
    if (!ir_host.valid()) die("ir_host says it's not valid");

    Array<real,8,memHost,styleFortran> wrap(ir_host,{2,2,2,2,2,2,2,2});
    if (wrap.extent(1) != test8d.extent(1)) die("wrap extent is wrong");
    if (wrap.data() != ir_host.data()) die("wrap data pointer is wrong");
    if (wrap.get_lbounds()(1) != 2) die("wrap wrong lbounds");


    ///////////////////////////////////////////////////////////
    // Test reshape
    ///////////////////////////////////////////////////////////
    auto reshaped = test8d.reshape({2,21},16,{-1,1132});
    reshaped = 2.f;
    if (yakl::intrinsics::sum(test8d) != d1*d2*d3*d4*d5*d6*d7*d8*2) { die("SimpleBounds: wrong sum for reshaped test8d"); }


    ///////////////////////////////////////////////////////////
    // Test collapse
    ///////////////////////////////////////////////////////////
    auto collapsed = test8d.collapse(-1);
    collapsed = 3.f;
    if (yakl::intrinsics::sum(test8d) != d1*d2*d3*d4*d5*d6*d7*d8*3) { die("SimpleBounds: wrong sum for collapsed test8d"); }

    auto constHostArr = construct_const_array_host();
    constHostArr.deallocate();
    if (constHostArr.initialized()) die("constHostArr: array didn't deallocate properly");

    auto constDevArr = construct_const_array_device();
    constDevArr.deallocate();
    if (constDevArr.initialized()) die("constDevArr: array didn't deallocate properly");

    ///////////////////////////////////////////////////////////
    // Test subset_slowest_dimension
    ///////////////////////////////////////////////////////////
    test7d = 1;
    test7d.subset_slowest_dimension(2) = 2;
    if (yakl::intrinsics::sum(test7d) != d1*d2*d3*d4*d5*d6*5) { die("SimpleBounds: wrong sum for reshaped subset"); }

    {
      yakl::Array<int,1,memHost,styleFortran> indices("indices",100);
      for (int i=1; i <= 100; i++) { indices(i) = i; }
      std::shuffle( indices.begin() , indices.end() , std::default_random_engine(13) );
      int tot = 0;
      for (int i=1; i <= 99; i++) { tot += std::abs( indices(i+1) - indices(i) ); }
      if (tot == 99) die("ERROR: Shuffle did not work for FArray");
    }

    {
      yakl::FSArray<int,1,yakl::SB<100>> indices;
      for (int i=1; i <= 100; i++) { indices(i) = i; }
      std::shuffle( indices.begin() , indices.end() , std::default_random_engine(13) );
      int tot = 0;
      for (int i=1; i <= 99; i++) { tot += std::abs( indices(i+1) - indices(i) ); }
      if (tot == 99) die("ERROR: Shuffle did not work for FArray");
    }

  }
  yakl::finalize();
  Kokkos::finalize(); 
  
  return 0;
}

