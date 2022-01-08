
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

typedef float real;

typedef Array<real,1,memHost,styleC> realHost1d;
typedef Array<real,2,memHost,styleC> realHost2d;
typedef Array<real,3,memHost,styleC> realHost3d;
typedef Array<real,4,memHost,styleC> realHost4d;
typedef Array<real,5,memHost,styleC> realHost5d;
typedef Array<real,6,memHost,styleC> realHost6d;
typedef Array<real,7,memHost,styleC> realHost7d;
typedef Array<real,8,memHost,styleC> realHost8d;

typedef Array<real,1,memDevice,styleC> real1d;
typedef Array<real,2,memDevice,styleC> real2d;
typedef Array<real,3,memDevice,styleC> real3d;
typedef Array<real,4,memDevice,styleC> real4d;
typedef Array<real,5,memDevice,styleC> real5d;
typedef Array<real,6,memDevice,styleC> real6d;
typedef Array<real,7,memDevice,styleC> real7d;
typedef Array<real,8,memDevice,styleC> real8d;

void die(std::string msg) {
  std::cerr << msg << std::endl;
  exit(-1);
}


Array<real const,1,memHost,styleC> construct_const_array_host() {
  return Array<real const,1,memHost,styleC>( realHost1d("arr",10) );
}


Array<real const,1,memDevice,styleC> construct_const_array_device() {
  return Array<real const,1,memDevice,styleC>( real1d("arr",10) );
}


int main() {
  yakl::init();
  {
    int constexpr d1 = 2;
    int constexpr d2 = 3;
    int constexpr d3 = 4;
    int constexpr d4 = 5;
    int constexpr d5 = 6;
    int constexpr d6 = 7;
    int constexpr d7 = 8;
    int constexpr d8 = 9;

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

    yakl::memset(test1d,0.f);
    yakl::memset(test2d,0.f);
    yakl::memset(test3d,0.f);
    yakl::memset(test4d,0.f);
    yakl::memset(test5d,0.f);
    yakl::memset(test6d,0.f);
    yakl::memset(test7d,0.f);
    yakl::memset(test8d,0.f);

    parallel_for( Bounds<1>(d1) , YAKL_LAMBDA (int i1) {
      test1d(i1) = 1;
    });
    parallel_for( Bounds<2>(d1,d2) , YAKL_LAMBDA (int i1, int i2) {
      test2d(i1,i2) = 1;
    });
    parallel_for( Bounds<3>(d1,d2,d3) , YAKL_LAMBDA (int i1, int i2, int i3) {
      test3d(i1,i2,i3) = 1;
    });
    parallel_for( Bounds<4>(d1,d2,d3,d4) , YAKL_LAMBDA (int i1, int i2, int i3, int i4) {
      test4d(i1,i2,i3,i4) = 1;
    });
    parallel_for( Bounds<5>(d1,d2,d3,d4,d5) , YAKL_LAMBDA (int i1, int i2, int i3, int i4, int i5) {
      test5d(i1,i2,i3,i4,i5) = 1;
    });
    parallel_for( Bounds<6>(d1,d2,d3,d4,d5,d6) , YAKL_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6) {
      test6d(i1,i2,i3,i4,i5,i6) = 1;
    });
    parallel_for( Bounds<7>(d1,d2,d3,d4,d5,d6,d7) , YAKL_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6, int i7) {
      test7d(i1,i2,i3,i4,i5,i6,i7) = 1;
    });
    parallel_for( Bounds<8>(d1,d2,d3,d4,d5,d6,d7,d8) , YAKL_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8) {
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

    yakl::memset(test1d_ptr,0.f);
    yakl::memset(test2d_ptr,0.f);
    yakl::memset(test3d_ptr,0.f);
    yakl::memset(test4d_ptr,0.f);
    yakl::memset(test5d_ptr,0.f);
    yakl::memset(test6d_ptr,0.f);
    yakl::memset(test7d_ptr,0.f);
    yakl::memset(test8d_ptr,0.f);

    parallel_for( Bounds<1>(d1) , YAKL_LAMBDA (int i1) {
      test1d_ptr(i1) = 1;
    });
    parallel_for( Bounds<2>(d1,d2) , YAKL_LAMBDA (int i1, int i2) {
      test2d_ptr(i1,i2) = 1;
    });
    parallel_for( Bounds<3>(d1,d2,d3) , YAKL_LAMBDA (int i1, int i2, int i3) {
      test3d_ptr(i1,i2,i3) = 1;
    });
    parallel_for( Bounds<4>(d1,d2,d3,d4) , YAKL_LAMBDA (int i1, int i2, int i3, int i4) {
      test4d_ptr(i1,i2,i3,i4) = 1;
    });
    parallel_for( Bounds<5>(d1,d2,d3,d4,d5) , YAKL_LAMBDA (int i1, int i2, int i3, int i4, int i5) {
      test5d_ptr(i1,i2,i3,i4,i5) = 1;
    });
    parallel_for( Bounds<6>(d1,d2,d3,d4,d5,d6) , YAKL_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6) {
      test6d_ptr(i1,i2,i3,i4,i5,i6) = 1;
    });
    parallel_for( Bounds<7>(d1,d2,d3,d4,d5,d6,d7) , YAKL_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6, int i7) {
      test7d_ptr(i1,i2,i3,i4,i5,i6,i7) = 1;
    });
    parallel_for( Bounds<8>(d1,d2,d3,d4,d5,d6,d7,d8) , YAKL_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8) {
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
    yakl::memset(testHost8d,0.f);
    if (yakl::intrinsics::sum(testHost8d) != 0) { die("memset: failed for testHost8d"); }

    ///////////////////////////////////////////////////////////
    // Test deep_copy_to device to host
    ///////////////////////////////////////////////////////////
    test8d.deep_copy_to(testHost8d);
    yakl::fence();
    if (yakl::intrinsics::sum(testHost8d) != d1*d2*d3*d4*d5*d6*d7*d8) { die("deep_copy_to: wrong sum for testHost8d"); }

    ///////////////////////////////////////////////////////////
    // Test device memset
    ///////////////////////////////////////////////////////////
    yakl::memset(test8d,0.f);
    if (yakl::intrinsics::sum(test8d) != 0) { die("memset: failed for test8d"); }

    ///////////////////////////////////////////////////////////
    // Test deep_copy_to host to device
    ///////////////////////////////////////////////////////////
    testHost8d.deep_copy_to(test8d);
    yakl::fence();
    if (yakl::intrinsics::sum(test8d) != d1*d2*d3*d4*d5*d6*d7*d8) { die("deep_copy_to: wrong sum for test8d"); }

    ///////////////////////////////////////////////////////////
    // Test createDeviceCopy from device
    ///////////////////////////////////////////////////////////
    auto test8d_dev2 = test8d.createDeviceCopy();
    if (yakl::intrinsics::sum(test8d_dev2) != d1*d2*d3*d4*d5*d6*d7*d8) { die("createDeviceCopy: wrong sum for test8d_dev2"); }

    ///////////////////////////////////////////////////////////
    // Test deep_copy_to device to device
    ///////////////////////////////////////////////////////////
    yakl::memset(test8d_dev2,0.f);
    test8d.deep_copy_to(test8d_dev2);
    yakl::fence();
    if (yakl::intrinsics::sum(test8d_dev2) != d1*d2*d3*d4*d5*d6*d7*d8) { die("deep_copy_to: wrong sum for test8d_dev2"); }

    ///////////////////////////////////////////////////////////
    // Test slice
    ///////////////////////////////////////////////////////////
    yakl::memset(test8d,0.f);
    auto slice = test8d.slice<3>(1,2,3,4,5,COLON,COLON,COLON);
    yakl::memset(slice,1.f);
    if (yakl::intrinsics::sum(test8d) != d6*d7*d8) { die("slice: wrong sum for slice"); }

    ///////////////////////////////////////////////////////////
    // Test slice inside a kernel
    ///////////////////////////////////////////////////////////
    yakl::memset(test8d,0.f);
    parallel_for( 1 , YAKL_LAMBDA (int dummy) {
      auto slice = test8d.slice<3>(1,2,3,4,5,COLON,COLON,COLON);
    });

    ///////////////////////////////////////////////////////////
    // Test non-standard loop bounds
    ///////////////////////////////////////////////////////////
    yakl::memset(test3d,0.);
    parallel_for( Bounds<3>(d1,{-1,d2-3},{0,d3,2}) , YAKL_LAMBDA (int i, int j, int k) {
      test3d(i,j+2,k) = 1;
    });
    if (yakl::intrinsics::sum(test3d) != 8) { die("non-standard loop: wrong sum for test3d");}
    

    ///////////////////////////////////////////////////////////
    // Test SimpleBounds
    ///////////////////////////////////////////////////////////
    yakl::memset(test1d,0.f);
    yakl::memset(test2d,0.f);
    yakl::memset(test3d,0.f);
    yakl::memset(test4d,0.f);
    yakl::memset(test5d,0.f);
    yakl::memset(test6d,0.f);
    yakl::memset(test7d,0.f);
    yakl::memset(test8d,0.f);

    parallel_for( SimpleBounds<1>(d1) , YAKL_LAMBDA (int i1) {
      test1d(i1) = 1;
    });
    parallel_for( SimpleBounds<2>(d1,d2) , YAKL_LAMBDA (int i1, int i2) {
      test2d(i1,i2) = 1;
    });
    parallel_for( SimpleBounds<3>(d1,d2,d3) , YAKL_LAMBDA (int i1, int i2, int i3) {
      test3d(i1,i2,i3) = 1;
    });
    parallel_for( SimpleBounds<4>(d1,d2,d3,d4) , YAKL_LAMBDA (int i1, int i2, int i3, int i4) {
      test4d(i1,i2,i3,i4) = 1;
    });
    parallel_for( SimpleBounds<5>(d1,d2,d3,d4,d5) , YAKL_LAMBDA (int i1, int i2, int i3, int i4, int i5) {
      test5d(i1,i2,i3,i4,i5) = 1;
    });
    parallel_for( SimpleBounds<6>(d1,d2,d3,d4,d5,d6) , YAKL_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6) {
      test6d(i1,i2,i3,i4,i5,i6) = 1;
    });
    parallel_for( SimpleBounds<7>(d1,d2,d3,d4,d5,d6,d7) , YAKL_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6, int i7) {
      test7d(i1,i2,i3,i4,i5,i6,i7) = 1;
    });
    parallel_for( SimpleBounds<8>(d1,d2,d3,d4,d5,d6,d7,d8) , YAKL_LAMBDA (int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8) {
      test8d(i1,i2,i3,i4,i5,i6,i7,i8) = 1;
    });

    if (yakl::intrinsics::sum(test1d) != d1                     ) { die("SimpleBounds: wrong sum for test1d"); }
    if (yakl::intrinsics::sum(test2d) != d1*d2                  ) { die("SimpleBounds: wrong sum for test2d"); }
    if (yakl::intrinsics::sum(test3d) != d1*d2*d3               ) { die("SimpleBounds: wrong sum for test3d"); }
    if (yakl::intrinsics::sum(test4d) != d1*d2*d3*d4            ) { die("SimpleBounds: wrong sum for test4d"); }
    if (yakl::intrinsics::sum(test5d) != d1*d2*d3*d4*d5         ) { die("SimpleBounds: wrong sum for test5d"); }
    if (yakl::intrinsics::sum(test6d) != d1*d2*d3*d4*d5*d6      ) { die("SimpleBounds: wrong sum for test6d"); }
    if (yakl::intrinsics::sum(test7d) != d1*d2*d3*d4*d5*d6*d7   ) { die("SimpleBounds: wrong sum for test7d"); }
    if (yakl::intrinsics::sum(test8d) != d1*d2*d3*d4*d5*d6*d7*d8) { die("SimpleBounds: wrong sum for test8d"); }


    ///////////////////////////////////////////////////////////
    // Test reshape
    ///////////////////////////////////////////////////////////
    auto reshaped = test8d.reshape<3>({20,16,1134});
    memset(reshaped,2.f);
    if (yakl::intrinsics::sum(test8d) != d1*d2*d3*d4*d5*d6*d7*d8*2) { die("SimpleBounds: wrong sum for reshaped test8d"); }


    ///////////////////////////////////////////////////////////
    // Test collapse
    ///////////////////////////////////////////////////////////
    auto collapsed = test8d.collapse();
    memset(collapsed,3.f);
    if (yakl::intrinsics::sum(test8d) != d1*d2*d3*d4*d5*d6*d7*d8*3) { die("SimpleBounds: wrong sum for collapsed test8d"); }

    auto constHostArr = construct_const_array_host();
    constHostArr.deallocate();
    if (constHostArr.initialized()) die("constHostArr: array didn't deallocate properly");

    auto constDevArr = construct_const_array_device();
    constDevArr.deallocate();
    if (constDevArr.initialized()) die("constDevArr: array didn't deallocate properly");

  }
  yakl::finalize();
  
  return 0;
}

