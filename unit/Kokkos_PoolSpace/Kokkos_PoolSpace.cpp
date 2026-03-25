
#include "YAKL.h"

void die(std::string msg) {
  Kokkos::abort(msg.c_str());
}



template <typename T, int N> struct KokkosType { using type = typename KokkosType<T*,N-1>::type; };
template <typename T> struct KokkosType<T,0> { using type = T; };

template <typename T> struct ViewArrayAnalysis {
  using base_type  = T;
  using value_type = T;
  static constexpr int rank = 0;
};
template <typename T> struct ViewArrayAnalysis<T*> {
  using base_type  = typename ViewArrayAnalysis<T>::base_type;
  using value_type = T*;
  static constexpr int rank = ViewArrayAnalysis<T>::rank + 1;
};



template <class KT, class MemSpace>
class ViewY : public Kokkos::View<KT,Kokkos::LayoutRight,MemSpace> {
  public:
  using base_t = Kokkos::View<KT,Kokkos::LayoutRight,MemSpace>;
  using base_t::base_t;
  using base_t::operator=;
  base_t const & get_View() const { return static_cast<base_t const &>(*this); }
  base_t       & get_View()       { return static_cast<base_t       &>(*this); }

  template <class TLOC> requires std::is_arithmetic_v<TLOC>
  ViewY const & operator=(TLOC const & v) const { Kokkos::deep_copy(*this,v); return *this; }

  template <class MemSpaceLoc>
  auto clone() const {
    auto func = [this] <std::size_t... Is> (std::index_sequence<Is...>) {
      return ViewY<typename base_t::non_const_data_type,MemSpaceLoc>( this->label() , this->extent(Is)... );
    };
    return func(std::make_index_sequence<base_t::rank>{});
  }

  template <class... ILOC> requires (std::is_integral_v<ILOC> && ...)
  KOKKOS_INLINE_FUNCTION auto slice(ILOC... slices) const {
    int constexpr nslice = sizeof...(ILOC);
    static_assert(nslice <= base_t::rank,"ERROR: too many indices in slice");
    int constexpr remaining = base_t::rank - nslice;
    using new_kt = typename KokkosType<typename base_t::value_type,remaining>::type;
    auto func = [&] < std::size_t... Is , std::size_t... Ir > ( std::index_sequence<Is...> , 
                                                                std::index_sequence<Ir...> ) {
      size_t offset = 0;
      ((offset += static_cast<size_t>(slices) * this->stride(Is)), ...);
      return ViewY<new_kt,MemSpace>( this->data()+offset , this->extent(nslice + Ir)... );
    };
    return func(std::make_index_sequence<nslice>{},std::make_index_sequence<remaining>{});
  }

  template <class... ILOC> requires (std::is_integral_v<ILOC> && ...)
  KOKKOS_INLINE_FUNCTION auto reshape(ILOC... newdims) const {
    int constexpr new_rank = sizeof...(ILOC);
    using new_kt = typename KokkosType<typename base_t::value_type,new_rank>::type;
    if ((static_cast<size_t>(newdims) * ...) != this->size()) {
      Kokkos::abort("ERROR: Resizing array with different total size");
    }
    return ViewY<new_kt,MemSpace>(this->data(),newdims...);
  }

  KOKKOS_INLINE_FUNCTION auto collapse() const {
    return ViewY<typename KokkosType<typename base_t::value_type,1>::type,MemSpace>(this->data(),this->size());
  }

  template <class ViewType>
  void deep_copy_to(ViewType const & them) const {
    Kokkos::deep_copy(them,*this);
    if constexpr (std::is_same_v<typename ViewType::memory_space,Kokkos::HostSpace>) Kokkos::fence();
  }

  auto createDeviceObject() const { return clone<yakl::PoolSpace>(); }

  auto createHostObject() const { return clone<Kokkos::HostSpace>(); }

  auto createDeviceCopy() const {
    auto ret = createDeviceObject();
    Kokkos::deep_copy( ret , *this );
    return ret;
  }

  auto createHostCopy() const {
    auto ret = createHostObject();
    Kokkos::deep_copy( ret , *this );
    Kokkos::fence();
    return ret;
  }

  KOKKOS_INLINE_FUNCTION base_t::value_type * begin() const { return this->data(); }

  KOKKOS_INLINE_FUNCTION base_t::value_type * end() const { return this->data()+this->size(); }

  inline friend std::ostream &operator<<( std::ostream& os , ViewY const & v ) {
    auto loc = v.createHostCopy(); // cout,cerr is expensive, so just create a host copy
    os << "Array [" << loc.label() << "], Dimensions [";
    for (int i = 0; i < loc.rank(); i++) { os << loc.extent(i) << (i<loc.rank()-1 ? "," : ""); }
    os << "] = " << loc.size() << " Elements:  ";
    for (int i = 0; i < loc.size(); i++) { os << loc.data()[i] << (i<loc.size()-1 ? " , " : ""); }
    os << std::endl;
    return os;
  }
};



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
    ViewY<float *,yakl::PoolSpace> arr("arr",10);
    Kokkos::parallel_for( "mykernel" , 10 , KOKKOS_LAMBDA (int i) {
      arr(i) = i+1;
    });
    auto arr_h = arr.createHostObject();
    arr.deep_copy_to(arr_h);
    auto arr_d = create_device_copy(arr_h);
    std::cout << sum(arr  ) << std::endl;
    std::cout << sum(arr_h) << std::endl;
    std::cout << sum(arr_d) << std::endl;
    ViewY<float[10],yakl::PoolSpace> arr_s("arr_s");
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
    auto tmp1 = arr.reshape(5,2);
    auto tmp2 = tmp1.slice(1);
    auto tmp3 = tmp2.createHostCopy();
    std::cout << tmp3;
    std::cout << "reshp use ct: " << arr.reshape(5,2).use_count() << std::endl;
  }
  yakl::finalize();
  Kokkos::finalize(); 
  return 0;
}

