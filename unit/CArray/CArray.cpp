
#include <iostream>
#include "YAKL.h"

using yakl::Array;
using yakl::styleC;
using yakl::memHost;
using yakl::memDevice;
using yakl::c::parallel_for;
using yakl::c::Bounds;
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

typedef Array<real,1,memHost,styleC> real1d;
typedef Array<real,2,memHost,styleC> real2d;
typedef Array<real,3,memHost,styleC> real3d;
typedef Array<real,4,memHost,styleC> real4d;
typedef Array<real,5,memHost,styleC> real5d;
typedef Array<real,6,memHost,styleC> real6d;
typedef Array<real,7,memHost,styleC> real7d;
typedef Array<real,8,memHost,styleC> real8d;


void die(std::string msg) {
  std::cerr << msg << std::endl;
  exit(-1);
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
    if (yakl::intrinsics::sum(test8d_dev2) != d1*d2*d3*d4*d5*d6*d7*d8) { die("deep_copy_to: wrong sum for test8d_dev2"); }

    ///////////////////////////////////////////////////////////
    // Test slice
    ///////////////////////////////////////////////////////////
    auto slice = test8d.slice<3>(1,2,3,4,5,COLON,COLON,COLON);
    if (yakl::intrinsics::sum(slice) != d6*d7*d8) { die("slice: wrong sum for slice"); }
  }
  yakl::finalize();
  
  return 0;
}

