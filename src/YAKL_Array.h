
#pragma once
// Included by YAKL_Array.h

namespace yakl {


  inline int constexpr COLON = 0;


  template <class Type> inline constexpr bool is_Array = requires { Type::is_Array; };



  template <typename T, int N> struct KokkosType { using type = typename KokkosType<T*,N-1>::type; };
  template <typename T> struct KokkosType<T,0> { using type = T; };



  template <class KT, class MemSpace>
  class Array : public Kokkos::View<KT,Kokkos::LayoutRight,MemSpace> {
    public:

    using base_t = Kokkos::View<KT,Kokkos::LayoutRight,MemSpace>;
    using this_t = Array<KT,MemSpace>;
    using base_t::base_t; 
    using base_t::operator=;
    using base_t::operator();

    bool static constexpr is_Array  = true ;
    bool static constexpr is_fstyle = false;
    bool static constexpr is_cstyle = true ;


    template <class TLOC> requires std::is_arithmetic_v<TLOC>
    Array const & operator=(TLOC const & v) const {
      Kokkos::deep_copy(*this,v);
      if constexpr (yakl_auto_fence) Kokkos::fence();
      return *this;
    }


    template <class MemSpaceLoc = MemSpace, class ValTypeLoc = typename base_t::non_const_value_type>
    auto clone_object() const {
      return [&] <std::size_t... Is> (std::index_sequence<Is...>) {
        using vtype = typename std::remove_cv_t<typename KokkosType<ValTypeLoc,base_t::rank()>::type>;
        return Array<vtype,MemSpaceLoc>( this->label() , this->extent(Is)... );
      } (std::make_index_sequence<this_t::rank()>{});
    }


    template <std::integral auto new_rank> requires (new_rank <= this_t::rank()) && (new_rank >= 0)
    KOKKOS_INLINE_FUNCTION auto slice(std::integral auto... indices) const requires (sizeof...(indices) == this_t::rank()) {
      int constexpr rank      = this_t::rank();
      int constexpr nslice    = rank - new_rank;
      int constexpr remaining = new_rank;
      using new_kt = typename KokkosType<typename base_t::value_type,remaining>::type;
      std::array<size_t,rank> slice_arr = { static_cast<size_t>(indices)... };
      size_t offset = 0;
      for (int i=0; i < nslice; i++) { offset += slice_arr[i] * this_t::stride(i); }
      return [&] <std::size_t... Ir> ( std::index_sequence<Ir...> ) {
        return Array<new_kt,MemSpace>( this_t::data()+offset , this_t::extent(nslice + Ir)... );
      } ( std::make_index_sequence<remaining>{} );
    }


    KOKKOS_INLINE_FUNCTION auto subset_slowest_dimension(std::integral auto u) const {
      return this_t::subset_slowest_dimension(0,u-1);
    }


    KOKKOS_INLINE_FUNCTION auto subset_slowest_dimension(std::integral auto l, std::integral auto u) const {
      auto constexpr rank = this_t::rank();
      typename this_t::value_type * offset = this_t::data()+l*this_t::stride(0);
      return [&] <std::size_t... Is> (std::index_sequence<Is...>) {
        return this_t( offset , u-l+1 , this->extent(1+Is)... );
      } (std::make_index_sequence<this_t::rank()-1>{});
    }


    KOKKOS_INLINE_FUNCTION auto reshape(std::integral auto... newdims) const {
      int constexpr new_rank = sizeof...(newdims);
      using new_kt = typename KokkosType<typename base_t::value_type,new_rank>::type;
      if ((static_cast<size_t>(newdims) * ...) != this_t::size()) {
        Kokkos::abort("ERROR: Resizing array with different total size");
      }
      return Array<new_kt,MemSpace>(this_t::data(),newdims...);
    }


    KOKKOS_INLINE_FUNCTION auto collapse() const {
      return Array<typename base_t::value_type *,MemSpace>(this_t::data(),this_t::size());
    }


    template <class ViewType>
    void deep_copy_to(ViewType const & them) const {
      if (them.size() != this->size()) Kokkos::abort("ERROR: calling deep_copy_to between differently sized arrays");
      Kokkos::deep_copy(them,*this);
      if constexpr (yakl_auto_fence) Kokkos::fence();
      if constexpr (std::is_same_v<typename ViewType::memory_space,Kokkos::HostSpace>) Kokkos::fence();
    }


    auto createDeviceObject() const { return clone_object<yakl::DeviceSpace>(); }


    auto createHostObject() const { return clone_object<Kokkos::HostSpace>(); }


    auto createDeviceCopy() const {
      auto ret = createDeviceObject();
      Kokkos::deep_copy( ret , *this );
      if constexpr (yakl_auto_fence) Kokkos::fence();
      return ret;
    }


    auto createHostCopy() const {
      auto ret = createHostObject();
      Kokkos::deep_copy( ret , *this );
      if constexpr (yakl_auto_fence) Kokkos::fence();
      Kokkos::fence();
      return ret;
    }


    template <class scalar_t> requires std::is_arithmetic_v<scalar_t>
    auto as() const {
      auto func = [&] <std::size_t... Is> (std::index_sequence<Is...>) {
        return Array<typename KokkosType<scalar_t,this_t::rank()>::type,MemSpace>( this->label() , this->extent(Is)... );
      };
      auto ret = func(std::make_index_sequence<this_t::rank()>{});
      YAKL_SCOPE( me , *this );
      Kokkos::parallel_for( "yakl_as_copy" ,
                            Kokkos::RangePolicy<typename base_t::execution_space>(0,this->size()) ,
                            KOKKOS_LAMBDA (int i) {
        ret.data()[i] = me.data()[i];
      });
      if constexpr (yakl_auto_fence) Kokkos::fence();
      return ret;
    }


    KOKKOS_INLINE_FUNCTION auto extents() const {
      SArray<size_t,this_t::rank()> ret;
      for (int i=0; i < this_t::rank(); i++) { ret(i) = this_t::extent(i); }
      return ret;
    }


    KOKKOS_INLINE_FUNCTION auto ubounds() const {
      SArray<size_t,this_t::rank()> ret;
      for (int i=0; i < this_t::rank(); i++) { ret(i) = this_t::extent(i)-1; }
      return ret;
    }


    KOKKOS_INLINE_FUNCTION auto lbounds() const {
      SArray<size_t,this_t::rank()> ret;
      for (int i=0; i < this_t::rank(); i++) { ret(i) = 0; }
      return ret;
    }


    KOKKOS_INLINE_FUNCTION base_t::value_type * begin() const { return this_t::data(); }
    KOKKOS_INLINE_FUNCTION base_t::value_type * end  () const { return this_t::data()+this_t::size(); }


    inline friend std::ostream &operator<<( std::ostream& os , Array const & v ) {
      auto loc = v.createHostCopy(); // cout,cerr is expensive, so just create a host copy
      os << "Array [" << loc.label() << "], Dimensions [";
      for (int i = 0; i < loc.rank(); i++) { os << loc.extent(i) << (i<loc.rank()-1 ? "," : ""); }
      os << "] = " << loc.size() << " Elements:  ";
      for (int i = 0; i < loc.size(); i++) { os << loc.data()[i] << (i<loc.size()-1 ? " , " : ""); }
      os << std::endl;
      return os;
    }


    KOKKOS_INLINE_FUNCTION auto unpack_global_index(size_t iglob) const {
      SArray<size_t,this_t::rank()> ret;
      for (int i=0; i < this_t::rank(); i++) { ret(i) = iglob / this_t::stride(i); }
      return ret;
    }


    base_t const & get_View() const { return static_cast<base_t const &>(*this); }
    base_t       & get_View()       { return static_cast<base_t       &>(*this); }
  };

}


