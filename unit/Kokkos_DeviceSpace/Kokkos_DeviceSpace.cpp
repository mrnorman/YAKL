
#include "YAKL.h"

// template <typename T> struct ViewArrayAnalysis {
//   using base_type  = T;
//   using value_type = T;
//   static constexpr int rank = 0;
// };
// template <typename T> struct ViewArrayAnalysis<T*> {
//   using base_type  = typename ViewArrayAnalysis<T>::base_type;
//   using value_type = T*;
//   static constexpr int rank = ViewArrayAnalysis<T>::rank + 1;
// };

using yakl::Array;
using yakl::Array_F;
using yakl::SArray;
using yakl::SArray_F;
using yakl::Bnds;
using yakl::Bounds;
using yakl::Bounds_F;
using yakl::parallel_for;
using yakl::parallel_for_F;
using yakl::intrinsics::sum;
using yakl::COLON;


int main() {
  Kokkos::initialize();
  yakl::init();
  {
    Array<float *,yakl::DeviceSpace> arr("arr",10);
    parallel_for( "mykernel" , 10 , KOKKOS_LAMBDA (int i) {
      arr(i) = i+1;
    });
    auto arr_h = arr.createHostObject();
    arr.deep_copy_to(arr_h);
    auto arr_d = arr_h.createDeviceCopy();
    std::cout << sum(arr  ) << std::endl;
    std::cout << sum(arr_h) << std::endl;
    std::cout << sum(arr_d) << std::endl;
    Array<float[10],yakl::DeviceSpace> arr_s("arr_s");
    arr_s = 1;
    std::cout << sum(arr_s) << std::endl;
    std::cout << "arr Rank:   " << arr.rank              () << std::endl;
    std::cout << "arr Size:   " << arr.size              () << std::endl;
    std::cout << "arr ptr:    " << arr.data              () << std::endl;
    std::cout << "arr begin:  " << arr.begin             () << std::endl;
    std::cout << "arr end:    " << arr.end               () << std::endl;
    std::cout << "arr contig: " << arr.span_is_contiguous() << std::endl;
    std::cout << "arr alloc:  " << arr.is_allocated      () << std::endl;
    std::cout << "arr alloc:  " << arr.label             () << std::endl;
    std::cout << "arr use ct: " << arr.use_count         () << std::endl;
    std::cout << arr;
    std::cout << arr.reshape(5,2);
    std::cout << arr.reshape(5,2).collapse();
    std::cout << arr.reshape(5,2).slice<1>(1,COLON);
    std::cout << "reshp use ct: " << arr.reshape(5,2).use_count() << std::endl;
    std::cout << arr.as<double>();
    std::cout << "arr.reshape(2,5) extents: " << arr.reshape(2,5).extents();
    std::cout << "arr.reshape(2,5) lbounds: " << arr.reshape(2,5).lbounds();
    std::cout << "arr.reshape(2,5) ubounds: " << arr.reshape(2,5).ubounds();
    SArray<float,3,2> csarray;
    csarray = 2;
    csarray(2,1) = 1;
    std::cout << csarray;
    std::cout << csarray.extents();
    std::cout << csarray.lbounds();
    std::cout << csarray.ubounds();
    std::cout << sum(csarray) << std::endl;
    SArray_F<float,Bnds{1,3},Bnds{1,2}> fsarray;
    fsarray = 2;
    fsarray(3,2) = 1;
    std::cout << fsarray;
    std::cout << fsarray.extents();
    std::cout << fsarray.lbounds();
    std::cout << fsarray.ubounds();
    std::cout << sum(fsarray) << std::endl;
    Array_F<float ***,yakl::DeviceSpace> farr("arr",{1,3},{1,3},{1,3});
    farr = 3;
    parallel_for_F( YAKL_AUTO_LABEL() , 1 , KOKKOS_LAMBDA (int i) { farr(1,1,1) = 1; });
    std::cout << farr.lbounds();
    std::cout << farr.ubounds();
    auto farr_re = farr.reshape(27);
    std::cout << farr_re.is_fstyle << " , " << farr_re.rank << std::endl;
    std::cout << farr_re.lbounds();
    std::cout << farr_re.ubounds();
    std::cout << sum(farr) << std::endl;
    parallel_for_F( YAKL_AUTO_LABEL() , 1 , KOKKOS_LAMBDA (int i) { farr_re(27) = 1; });
    Kokkos::fence();
    std::cout << farr_re;
    std::cout << sum(farr_re) << std::endl;
    Array_F<float *,yakl::DeviceSpace> farr2("farr2",{1,10});
    parallel_for_F( YAKL_AUTO_LABEL() , 11 , KOKKOS_LAMBDA (int i) {
      farr2(i) = i;
    });
    std::cout << farr2;
    std::cout << farr2.reshape(5,2).slice<1>(COLON,2);
    Bounds_F<3> bnds({1,3},{-1,5},{2,7,2});
  }
  yakl::finalize();
  Kokkos::finalize(); 
  return 0;
}

