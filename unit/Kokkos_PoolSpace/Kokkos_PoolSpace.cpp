
#include "YAKL.h"

void die(std::string msg) {
  Kokkos::abort(msg.c_str());
}


template <class ViewType>
inline void constexpr assert_contiguous() {
  static_assert( std::is_same_v<typename ViewType::array_layout,Kokkos::LayoutLeft > || 
                 std::is_same_v<typename ViewType::array_layout,Kokkos::LayoutRight> ,
                 "ERROR: summation assumes contiguity, LayoutLeft or LayoutRight"    );
}


template <class ViewType>
inline typename ViewType::non_const_value_type sum(ViewType const & a) {
  assert_contiguous<ViewType>();
  using scalar_t = typename ViewType::non_const_value_type;
  scalar_t result;
  Kokkos::parallel_reduce( "yakl_sum" ,
                           Kokkos::RangePolicy<typename ViewType::execution_space>(0,a.size()) ,
                           KOKKOS_LAMBDA (int i , scalar_t & lsum ) {
    lsum += a.data()[i];
  } , result );
  return result;
}


template <class ViewType> inline decltype(Kokkos::create_mirror(yakl::PoolSpace{},ViewType()))
create_device_object(ViewType const &in) {
  return Kokkos::create_mirror(yakl::PoolSpace{},in);
}


template <class ViewType> inline decltype(Kokkos::create_mirror(Kokkos::HostSpace{},ViewType()))
create_host_object(ViewType const &in) {
  return Kokkos::create_mirror(Kokkos::HostSpace{},in);
}


template <class ViewType> inline decltype(Kokkos::create_mirror(yakl::PoolSpace{},ViewType()))
create_device_copy(ViewType const &in) {
  auto out = Kokkos::create_mirror(yakl::PoolSpace{},in);
  Kokkos::deep_copy(out,in);
  return out;
}


template <class ViewType> inline decltype(Kokkos::create_mirror(Kokkos::HostSpace{},ViewType()))
create_host_copy(ViewType const &in) {
  auto out = Kokkos::create_mirror(Kokkos::HostSpace{},in);
  Kokkos::deep_copy(out,in);
  return out;
}


int main() {
  Kokkos::initialize();
  yakl::init();
  {
    Kokkos::View<float *,Kokkos::LayoutRight,yakl::PoolSpace> arr("arr",10);
    Kokkos::deep_copy(arr,1);
    auto arr_h = create_host_object(arr);
    Kokkos::deep_copy(arr_h,arr);
    auto arr_d = create_device_copy(arr_h);
    std::cout << sum(arr  ) << std::endl;
    std::cout << sum(arr_h) << std::endl;
    std::cout << sum(arr_d) << std::endl;
    Kokkos::View<float[10],Kokkos::LayoutRight,yakl::PoolSpace> arr_s("arr_s");
    Kokkos::deep_copy(arr_s,1);
    std::cout << sum(arr_s) << std::endl;
  }
  yakl::finalize();
  Kokkos::finalize(); 
  return 0;
}

