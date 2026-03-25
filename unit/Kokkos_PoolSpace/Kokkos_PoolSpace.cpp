
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
  auto clone_object() const {
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

  auto createDeviceObject() const { return clone_object<yakl::PoolSpace>(); }

  auto createHostObject() const { return clone_object<Kokkos::HostSpace>(); }

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

  template <class scalar_t> requires std::is_arithmetic_v<scalar_t>
  auto as() {
    auto func = [this] <std::size_t... Is> (std::index_sequence<Is...>) {
      return ViewY<typename KokkosType<scalar_t,base_t::rank>::type,MemSpace>( this->label() , this->extent(Is)... );
    };
    auto ret = func(std::make_index_sequence<base_t::rank>{});
    YAKL_SCOPE( me , *this );
    Kokkos::parallel_for( "yakl_as_copy" ,
                          Kokkos::RangePolicy<typename base_t::execution_space>(0,this->size()) ,
                          KOKKOS_LAMBDA (int i) {
      ret.data()[i] = me.data()[i];
    });
    return ret;
  }

  KOKKOS_INLINE_FUNCTION auto extents() const {
    yakl::CSArray<size_t,1,base_t::rank> ret;
    for (int i=0; i < base_t::rank; i++) { ret(i) = this->extent(i); }
    return ret;
  }

  KOKKOS_INLINE_FUNCTION auto ubounds() const {
    yakl::CSArray<size_t,1,base_t::rank> ret;
    for (int i=0; i < base_t::rank; i++) { ret(i) = this->extent(i)-1; }
    return ret;
  }

  KOKKOS_INLINE_FUNCTION auto lbounds() const {
    yakl::CSArray<size_t,1,base_t::rank> ret;
    for (int i=0; i < base_t::rank; i++) { ret(i) = 0; }
    return ret;
  }

  KOKKOS_INLINE_FUNCTION base_t::value_type * begin() const { return this->data(); }
  KOKKOS_INLINE_FUNCTION base_t::value_type * end  () const { return this->data()+this->size(); }

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



template <class T, std::size_t... DIMS> requires (sizeof...(DIMS) > 0)
class CSArray {
  public:
  bool                    static constexpr is_CSArray    = true;
  size_t                  static constexpr rank          = sizeof...(DIMS);
  size_t                  static constexpr num_elements  = (DIMS * ... * 1);
  size_t                  static constexpr dims[rank]    = {DIMS...};
  std::array<size_t,rank> static constexpr offsets       = [] {
    std::array<size_t,rank> result = {};
    for (int i=0; i < rank; i++) {
      result[i] = 1;
      for (int j = i+1; j < rank; j++) result[i] *= dims[j];
    }
    return result;
  }();
  using value_type           = T;
  using const_value_type     = typename std::add_const_t<T>;
  using non_const_value_type = typename std::remove_cv_t<T>;

  T mutable my_data[num_elements];

  template <class TLOC> requires std::is_arithmetic_v<TLOC>
  KOKKOS_INLINE_FUNCTION void operator= (TLOC val) { for (int i=0 ; i < size() ; i++) { my_data[i] = val; } }

  template <class... ILOC> requires (std::is_integral_v<ILOC> && ...)
  KOKKOS_INLINE_FUNCTION T & operator()(ILOC... indices) const {
    static_assert( sizeof...(ILOC) == rank , "ERROR: Indexing CSArray with the wrong number of indices" );
    size_t offset = 0;
    [&] <std::size_t... Is>(std::index_sequence<Is...>) {
        ((offset += static_cast<size_t>(indices) * offsets[Is]), ...);
    }(std::make_index_sequence<rank>{});
    #ifdef KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
      bool iob = false;
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
          ((iob = iob || static_cast<std::ptrdiff_t>(indices) < 0 ||
                         static_cast<size_t>(indices) >= dims[Is]), ...);
      }(std::make_index_sequence<rank>{});
      if (iob) Kokkos::abort("ERROR: CSArray index out of bounds");
    #endif
    return my_data[offset];
  }

  KOKKOS_INLINE_FUNCTION T * data () const { return my_data; }
  KOKKOS_INLINE_FUNCTION T * begin() const { return my_data; }
  KOKKOS_INLINE_FUNCTION T * end  () const { return my_data + size(); }
  KOKKOS_INLINE_FUNCTION size_t static constexpr size() { return num_elements; }
  KOKKOS_INLINE_FUNCTION bool   static constexpr span_is_contiguous() { return true; }
  template <class ILOC> requires std::is_integral_v<ILOC>
  KOKKOS_INLINE_FUNCTION size_t static constexpr extent(ILOC i) {
    #ifdef KOKKOS_ENABLE_DEBUG
      if (i < 0 || i > rank-1) Kokkos::abort("ERROR: calling CArray extent() with out of bounds index"); 
    #endif
    return dims[i];
  }

  inline friend std::ostream &operator<<( std::ostream& os , CSArray const & v ) {
    os << "yakl::CSArray: ";
    for (size_t i = 0; i < size(); i++) { os << v.my_data[i] << (i<size()-1 ? " , " : ""); }
    os << std::endl;
    return os;
  }

  KOKKOS_INLINE_FUNCTION auto extents() const {
    CSArray<size_t,rank> ret;
    for (int i=0; i < rank; i++) { ret(i) = dims[i]; }
    return ret;
  }

  KOKKOS_INLINE_FUNCTION auto lbounds() const {
    CSArray<size_t,rank> ret;
    for (int i=0; i < rank; i++) { ret(i) = 0; }
    return ret;
  }

  KOKKOS_INLINE_FUNCTION auto ubounds() const {
    CSArray<size_t,rank> ret;
    for (int i=0; i < rank; i++) { ret(i) = dims[i]-1; }
    return ret;
  }
};



template <class T> inline constexpr bool is_CSArray = requires { T::is_CSArray; };



template <class ViewType>
inline typename ViewType::non_const_value_type sum(ViewType const & a) {
  #ifdef KOKKOS_ENABLE_DEBUG
    if (!ViewType.span_is_contiguous()) Kokkos::abort("ERROR: Computing sum on non-contiguous View");
  #endif
  using scalar_t = typename ViewType::non_const_value_type;
  scalar_t result = 0;
  if constexpr (is_CSArray<ViewType>) {
    for (int i=0; i < a.size(); i++) { result += a.data()[i]; }
  } else {
    Kokkos::parallel_reduce( "yakl_sum" ,
                             Kokkos::RangePolicy<typename ViewType::execution_space>(0,a.size()) ,
                             KOKKOS_LAMBDA (int i , scalar_t & lsum ) {
      lsum += a.data()[i];
    } , result );
  }
  return result;
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
    auto arr_d = arr_h.createDeviceCopy();
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
    std::cout << arr.reshape(5,2).slice(1);
    std::cout << "reshp use ct: " << arr.reshape(5,2).use_count() << std::endl;
    std::cout << arr.as<double>();
    std::cout << "arr.reshape(2,5) extents: " << arr.reshape(2,5).extents();
    std::cout << "arr.reshape(2,5) lbounds: " << arr.reshape(2,5).lbounds();
    std::cout << "arr.reshape(2,5) ubounds: " << arr.reshape(2,5).ubounds();
    CSArray<float,3,2> nsarray;
    nsarray = 2;
    std::cout << nsarray;
    std::cout << nsarray.extents();
    std::cout << nsarray.lbounds();
    std::cout << nsarray.ubounds();
    std::cout << sum(nsarray) << std::endl;
  }
  yakl::finalize();
  Kokkos::finalize(); 
  return 0;
}

