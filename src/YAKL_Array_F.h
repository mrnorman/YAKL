
#pragma once
// Included by YAKL_Array.h

namespace yakl {

  template <class KT, class MemSpace = DeviceSpace>
  class Array_F : public Kokkos::View<KT,Kokkos::LayoutLeft,MemSpace> {
    public:

    using base_t = Kokkos::View<KT,Kokkos::LayoutLeft,MemSpace>;
    using this_t = Array_F<KT,MemSpace>;

    bool static constexpr is_Array  = true ;
    bool static constexpr is_fstyle = true ;
    bool static constexpr is_cstyle = false;
    bool static constexpr on_device = std::is_same_v<yakl::DeviceSpace,MemSpace>;

    std::array<ptrdiff_t,this_t::rank()> lb = {};

    private:

    template <class, class> friend class Array_F;

    template <class SourceView>
    void retain_reference_host(SourceView const & source, size_t offset) {
#if KOKKOS_VERSION >= 40700
      auto handle = source.accessor().offset(source.data_handle(),offset);
      base_t::operator=(base_t(handle,this->mapping()));
#else
      base_t::operator=(base_t(source.impl_track(),this->impl_map()));
#endif
    }

    template <class SourceView>
    void retain_subview_host(SourceView const & source) {
      base_t::operator=(source);
    }

    template <int NewRank, size_t I, size_t Rank>
    auto slice_argument_host(std::array<ptrdiff_t,Rank> const & indices) const {
      if constexpr (I < NewRank) return Kokkos::ALL;
      else                       return indices[I]-lb[I];
    }

    template <int NewRank, class Loc, size_t Rank, size_t... Is>
    void retain_slice_reference_host(Loc & loc, std::array<ptrdiff_t,Rank> const & indices,
                                     std::index_sequence<Is...>) const {
      auto subview = Kokkos::subview(static_cast<base_t const &>(*this),
                                     slice_argument_host<NewRank,Is>(indices)...);
      loc.retain_subview_host(subview);
    }

    KOKKOS_INLINE_FUNCTION static ptrdiff_t slice_index(std::integral auto index) {
      return static_cast<ptrdiff_t>(index);
    }

    KOKKOS_INLINE_FUNCTION static ptrdiff_t slice_index(Kokkos::ALL_t) { return 0; }

    public:


    // AB: ArrayBounds
    struct AB {
      ptrdiff_t l, u;
      KOKKOS_INLINE_FUNCTION AB(std::integral auto u) : l(1) , u(u) { }
      KOKKOS_INLINE_FUNCTION AB(std::integral auto l, std::integral auto u) : l(l) , u(u) { }
    };

    KOKKOS_INLINE_FUNCTION static size_t checked_extent(AB bnd) {
      using unsigned_bound_t = std::make_unsigned_t<ptrdiff_t>;
      if constexpr (kokkos_debug) {
        if (bnd.u < bnd.l) Kokkos::abort("ERROR: Array_F upper bound is less than its lower bound");
        auto const difference = static_cast<unsigned_bound_t>(bnd.u) - static_cast<unsigned_bound_t>(bnd.l);
        if (difference > std::numeric_limits<size_t>::max()-1) {
          Kokkos::abort("ERROR: Array_F extent overflow");
        }
      }
      auto const difference = static_cast<unsigned_bound_t>(bnd.u) - static_cast<unsigned_bound_t>(bnd.l);
      return static_cast<size_t>(difference) + 1;
    }

    KOKKOS_INLINE_FUNCTION static typename this_t::value_type * checked_pointer(typename this_t::value_type * ptr) {
      if constexpr (kokkos_debug) {
        if (ptr == nullptr) Kokkos::abort("ERROR: constructing a nonempty unmanaged Array_F from a null pointer");
      }
      return ptr;
    }


    Array_F() = default;
    ~Array_F() = default;


    // Owned constructors
    Array_F(std::string const & label, AB b1)
        requires (this_t::rank()==1)
        : base_t(label,checked_extent(b1)) ,
          lb({b1.l}) {}
    Array_F(std::string const & label, AB b1, AB b2)
        requires (this_t::rank()==2)
        : base_t(label,checked_extent(b1),checked_extent(b2)) ,
          lb({b1.l,b2.l}) {}
    Array_F(std::string const & label, AB b1, AB b2, AB b3)
        requires (this_t::rank()==3)
        : base_t(label,checked_extent(b1),checked_extent(b2),checked_extent(b3)) ,
          lb({b1.l,b2.l,b3.l}) {}
    Array_F(std::string const & label, AB b1, AB b2, AB b3, AB b4)
        requires (this_t::rank()==4)
        : base_t(label,checked_extent(b1),checked_extent(b2),checked_extent(b3),checked_extent(b4)) ,
          lb({b1.l,b2.l,b3.l,b4.l}) {}
    Array_F(std::string const & label, AB b1, AB b2, AB b3, AB b4, AB b5)
        requires (this_t::rank()==5)
        : base_t(label,checked_extent(b1),checked_extent(b2),checked_extent(b3),checked_extent(b4),checked_extent(b5)) ,
          lb({b1.l,b2.l,b3.l,b4.l,b5.l}) {}
    Array_F(std::string const & label, AB b1, AB b2, AB b3, AB b4, AB b5, AB b6)
        requires (this_t::rank()==6)
        : base_t(label,checked_extent(b1),checked_extent(b2),checked_extent(b3),checked_extent(b4),checked_extent(b5),
                 checked_extent(b6)) ,
          lb({b1.l,b2.l,b3.l,b4.l,b5.l,b6.l}) {}
    Array_F(std::string const & label, AB b1, AB b2, AB b3, AB b4, AB b5, AB b6, AB b7)
        requires (this_t::rank()==7)
        : base_t(label,checked_extent(b1),checked_extent(b2),checked_extent(b3),checked_extent(b4),checked_extent(b5),
                 checked_extent(b6),checked_extent(b7)) ,
          lb({b1.l,b2.l,b3.l,b4.l,b5.l,b6.l,b7.l}) {}
    Array_F(std::string const & label, AB b1, AB b2, AB b3, AB b4, AB b5, AB b6, AB b7, AB b8)
        requires (this_t::rank()==8)
        : base_t(label,checked_extent(b1),checked_extent(b2),checked_extent(b3),checked_extent(b4),checked_extent(b5),
                 checked_extent(b6),checked_extent(b7),checked_extent(b8)) ,
          lb({b1.l,b2.l,b3.l,b4.l,b5.l,b6.l,b7.l,b8.l}) {}
    // Non-owned constructors
    KOKKOS_INLINE_FUNCTION Array_F(typename this_t::value_type *ptr, AB b1)
        requires (this_t::rank()==1)
        : base_t(checked_pointer(ptr),checked_extent(b1)) ,
          lb({b1.l}) {}
    KOKKOS_INLINE_FUNCTION Array_F(typename this_t::value_type *ptr, AB b1, AB b2)
        requires (this_t::rank()==2)
        : base_t(checked_pointer(ptr),checked_extent(b1),checked_extent(b2)) ,
          lb({b1.l,b2.l}) {}
    KOKKOS_INLINE_FUNCTION Array_F(typename this_t::value_type *ptr, AB b1, AB b2, AB b3)
        requires (this_t::rank()==3)
        : base_t(checked_pointer(ptr),checked_extent(b1),checked_extent(b2),checked_extent(b3)) ,
          lb({b1.l,b2.l,b3.l}) {}
    KOKKOS_INLINE_FUNCTION Array_F(typename this_t::value_type *ptr, AB b1, AB b2, AB b3, AB b4)
        requires (this_t::rank()==4)
        : base_t(checked_pointer(ptr),checked_extent(b1),checked_extent(b2),checked_extent(b3),checked_extent(b4)) ,
          lb({b1.l,b2.l,b3.l,b4.l}) {}
    KOKKOS_INLINE_FUNCTION Array_F(typename this_t::value_type *ptr, AB b1, AB b2, AB b3, AB b4, AB b5)
        requires (this_t::rank()==5)
        : base_t(checked_pointer(ptr),checked_extent(b1),checked_extent(b2),checked_extent(b3),checked_extent(b4),
                 checked_extent(b5)) ,
          lb({b1.l,b2.l,b3.l,b4.l,b5.l}) {}
    KOKKOS_INLINE_FUNCTION Array_F(typename this_t::value_type *ptr, AB b1, AB b2, AB b3, AB b4, AB b5, AB b6)
        requires (this_t::rank()==6)
        : base_t(checked_pointer(ptr),checked_extent(b1),checked_extent(b2),checked_extent(b3),checked_extent(b4),checked_extent(b5),
                 checked_extent(b6)) ,
          lb({b1.l,b2.l,b3.l,b4.l,b5.l,b6.l}) {}
    KOKKOS_INLINE_FUNCTION Array_F(typename this_t::value_type *ptr, AB b1, AB b2, AB b3, AB b4, AB b5, AB b6, AB b7)
        requires (this_t::rank()==7)
        : base_t(checked_pointer(ptr),checked_extent(b1),checked_extent(b2),checked_extent(b3),checked_extent(b4),checked_extent(b5),
                 checked_extent(b6),checked_extent(b7)) ,
          lb({b1.l,b2.l,b3.l,b4.l,b5.l,b6.l,b7.l}) {}
    KOKKOS_INLINE_FUNCTION Array_F(typename this_t::value_type *ptr, AB b1, AB b2, AB b3, AB b4, AB b5, AB b6, AB b7, AB b8)
        requires (this_t::rank()==8)
        : base_t(checked_pointer(ptr),checked_extent(b1),checked_extent(b2),checked_extent(b3),checked_extent(b4),checked_extent(b5),
                 checked_extent(b6),checked_extent(b7),checked_extent(b8)) ,
          lb({b1.l,b2.l,b3.l,b4.l,b5.l,b6.l,b7.l,b8.l}) {}

    Array_F(Array_F const &) = default;
    Array_F(Array_F &&) = default;
    Array_F & operator=(Array_F const &) = default;
    Array_F & operator=(Array_F &&) = default;


    template <class KT_RHS, class MemSpace_RHS>
    KOKKOS_INLINE_FUNCTION Array_F(Array_F<KT_RHS,MemSpace_RHS> const & rhs)
          : lb(rhs.lb) , base_t(static_cast<Kokkos::View<KT_RHS,Kokkos::LayoutLeft,MemSpace_RHS>>(rhs)) { }
    template <class KT_RHS, class MemSpace_RHS>
    KOKKOS_INLINE_FUNCTION Array_F(Array_F<KT_RHS,MemSpace_RHS> && rhs)
          : lb(std::move(rhs.lb)) , base_t(static_cast<Kokkos::View<KT_RHS,Kokkos::LayoutLeft,MemSpace_RHS>>(std::move(rhs))) { }
    template <class KT_RHS, class MemSpace_RHS>
    KOKKOS_INLINE_FUNCTION Array_F & operator=(Array_F<KT_RHS,MemSpace_RHS> const & rhs) {
      lb = rhs.lb;
      base_t::operator=(static_cast<Kokkos::View<KT_RHS,Kokkos::LayoutLeft,MemSpace_RHS>>(rhs));
      return *this;
    }
    template <class KT_RHS, class MemSpace_RHS>
    KOKKOS_INLINE_FUNCTION Array_F & operator=(Array_F<KT_RHS,MemSpace_RHS> && rhs) {
      lb = std::move(rhs.lb);
      base_t::operator=(static_cast<Kokkos::View<KT_RHS,Kokkos::LayoutLeft,MemSpace_RHS>>(std::move(rhs)));
      return *this;
    }


    // Fortran-style operator()
    KOKKOS_INLINE_FUNCTION auto & operator()(std::integral auto i0) const requires (this_t::rank()==1) {
      return base_t::operator()(i0-this_t::lb[0]);
    }
    KOKKOS_INLINE_FUNCTION auto & operator()(std::integral auto i0,
                                             std::integral auto i1) const requires (this_t::rank()==2) {
      return base_t::operator()(i0-this_t::lb[0],
                                i1-this_t::lb[1]);
    }
    KOKKOS_INLINE_FUNCTION auto & operator()(std::integral auto i0,
                                             std::integral auto i1,
                                             std::integral auto i2) const requires (this_t::rank()==3) {
      return base_t::operator()(i0-this_t::lb[0],
                                i1-this_t::lb[1],
                                i2-this_t::lb[2]);
    }
    KOKKOS_INLINE_FUNCTION auto & operator()(std::integral auto i0,
                                             std::integral auto i1,
                                             std::integral auto i2,
                                             std::integral auto i3) const requires (this_t::rank()==4) {
      return base_t::operator()(i0-this_t::lb[0],
                                i1-this_t::lb[1],
                                i2-this_t::lb[2],
                                i3-this_t::lb[3]);
    }
    KOKKOS_INLINE_FUNCTION auto & operator()(std::integral auto i0,
                                             std::integral auto i1,
                                             std::integral auto i2,
                                             std::integral auto i3,
                                             std::integral auto i4) const requires (this_t::rank()==5) {
      return base_t::operator()(i0-this_t::lb[0],
                                i1-this_t::lb[1],
                                i2-this_t::lb[2],
                                i3-this_t::lb[3],
                                i4-this_t::lb[4]);
    }
    KOKKOS_INLINE_FUNCTION auto & operator()(std::integral auto i0,
                                             std::integral auto i1,
                                             std::integral auto i2,
                                             std::integral auto i3,
                                             std::integral auto i4,
                                             std::integral auto i5) const requires (this_t::rank()==6) {
      return base_t::operator()(i0-this_t::lb[0],
                                i1-this_t::lb[1],
                                i2-this_t::lb[2],
                                i3-this_t::lb[3],
                                i4-this_t::lb[4],
                                i5-this_t::lb[5]);
    }
    KOKKOS_INLINE_FUNCTION auto & operator()(std::integral auto i0,
                                             std::integral auto i1,
                                             std::integral auto i2,
                                             std::integral auto i3,
                                             std::integral auto i4,
                                             std::integral auto i5,
                                             std::integral auto i6) const requires (this_t::rank()==7) {
      return base_t::operator()(i0-this_t::lb[0],
                                i1-this_t::lb[1],
                                i2-this_t::lb[2],
                                i3-this_t::lb[3],
                                i4-this_t::lb[4],
                                i5-this_t::lb[5],
                                i6-this_t::lb[6]);
    }
    KOKKOS_INLINE_FUNCTION auto & operator()(std::integral auto i0,
                                             std::integral auto i1,
                                             std::integral auto i2,
                                             std::integral auto i3,
                                             std::integral auto i4,
                                             std::integral auto i5,
                                             std::integral auto i6,
                                             std::integral auto i7) const requires (this_t::rank()==8) {
      return base_t::operator()(i0-this_t::lb[0],
                                i1-this_t::lb[1],
                                i2-this_t::lb[2],
                                i3-this_t::lb[3],
                                i4-this_t::lb[4],
                                i5-this_t::lb[5],
                                i6-this_t::lb[6],
                                i7-this_t::lb[7]);
    }


    template <class TLOC> requires std::is_arithmetic_v<TLOC>
    Array_F const & operator=(TLOC const & v) const {
      if constexpr (kokkos_debug) {
        if (!this_t::is_allocated()) Kokkos::abort("ERROR: assigning a scalar to an unallocated Array_F");
      }
      if constexpr (yakl_auto_profile) timer_start("yakl::Array_F::operator=scalar");
      Kokkos::deep_copy(*this,v);
      if constexpr (yakl_auto_profile) timer_stop("yakl::Array_F::operator=scalar");
      if constexpr (yakl_auto_fence) Kokkos::fence();
      return *this;
    }


    template <class MemSpaceLoc = MemSpace, class ValTypeLoc = typename base_t::non_const_value_type>
    auto clone_object() const {
      return [&] <std::size_t... Is> (std::index_sequence<Is...>) {
        using vtype = typename std::remove_cv_t<typename ViewType<ValTypeLoc,base_t::rank()>::type>;
        return Array_F<vtype,MemSpaceLoc>( this->label() , {lb[Is],lb[Is]+this->extent(Is)-1}... );
      } (std::make_index_sequence<this_t::rank()>{});
    }


    template <std::integral auto new_rank, class... INTS> requires (new_rank <= this_t::rank()) &&
                                                                   (new_rank >= 0) &&
                                                                   ((std::integral<INTS> ||
                                                                     std::same_as<INTS,Kokkos::ALL_t>) && ...)
    KOKKOS_INLINE_FUNCTION auto slice(INTS... indices) const {
      static_assert(sizeof...(indices)==this_t::rank(),"ERROR: slice calld with wrong number of indices");
      int  constexpr rank      = this_t::rank();
      int  constexpr nslice    = rank - new_rank;
      int  constexpr remaining = new_rank;
      auto const    &lb        = this_t::lb;
      using new_kt = typename ViewType<typename base_t::value_type,remaining>::type;
      using index_types = std::tuple<INTS...>;
      static_assert([] <size_t... Is> (std::index_sequence<Is...>) {
        return (std::integral<std::tuple_element_t<remaining+Is,index_types>> && ...);
      } (std::make_index_sequence<nslice>{}), "ERROR: Array_F::slice requires an integer for each removed dimension");
      std::array<ptrdiff_t,rank> slice_arr = { slice_index(indices)... };
      if constexpr (kokkos_debug) {
        if (!this_t::is_allocated()) Kokkos::abort("ERROR: slicing an unallocated Array_F");
      }
      if constexpr (kokkos_bounds_debug) {
        for (int i=0; i < nslice; i++) {
          int const dim = rank-1-i;
          ptrdiff_t const ub = lb[dim] + static_cast<ptrdiff_t>(this_t::extent(dim)) - 1;
          if (slice_arr[dim] < lb[dim] || slice_arr[dim] > ub) {
            Kokkos::abort("ERROR: Array_F::slice index out of bounds");
          }
        }
      }
      size_t offset = 0;
      for (int i=0; i < nslice; i++) {
        offset += (slice_arr[rank-1-i]-lb[rank-1-i]) * this_t::stride(rank-1-i);
      }
      return [&] <std::size_t... Ir> ( std::index_sequence<Ir...> ) {
        auto loc = Array_F<new_kt,MemSpace>( this_t::data()+offset , this_t::extent(Ir)... );
        for (int i=0; i < remaining; i++) { loc.lb[i] = lb[i]; }
        KOKKOS_IF_ON_HOST(( retain_slice_reference_host<new_rank>(loc,slice_arr,std::make_index_sequence<rank>{}); ))
        return loc;
      } ( std::make_index_sequence<remaining>{} );
    }


    KOKKOS_INLINE_FUNCTION auto subset_slowest_dimension(std::integral auto u) const {
      return this_t::subset_slowest_dimension(this_t::lb[this_t::rank()-1],u);
    }


    KOKKOS_INLINE_FUNCTION auto subset_slowest_dimension(std::integral auto l, std::integral auto u) const {
      auto constexpr rank = this_t::rank();
      auto const    &lb   = this_t::lb;
      if constexpr (kokkos_debug) {
        if (!this_t::is_allocated()) Kokkos::abort("ERROR: subsetting an unallocated Array_F");
      }
      if constexpr (kokkos_bounds_debug) {
        ptrdiff_t const upper = lb[rank-1] + static_cast<ptrdiff_t>(this_t::extent(rank-1)) - 1;
        if (l < lb[rank-1] || u < l || u > upper) {
          Kokkos::abort("ERROR: Array_F::subset_slowest_dimension bounds are invalid");
        }
      }
      typename this_t::value_type * offset = this_t::data()+(l-lb[rank-1])*this_t::stride(rank-1);
      return [&] <std::size_t... Is> (std::index_sequence<Is...>) {
        auto loc = this_t( offset , {lb[Is],lb[Is]+this_t::extent(Is)-1}... , {l,u} );
        KOKKOS_IF_ON_HOST(( loc.retain_reference_host(*this,(l-lb[rank-1])*this_t::stride(rank-1)); ))
        return loc;
      } (std::make_index_sequence<this_t::rank()-1>{});
    }


    template <class... BNDS> requires (std::is_same_v<BNDS,AB> && ...)
    KOKKOS_INLINE_FUNCTION auto reshape_all(BNDS... newdims) const {
      int constexpr new_rank = sizeof...(newdims);
      using new_kt = typename ViewType<typename base_t::value_type,new_rank>::type;
      std::array<AB,new_rank> bounds = {newdims...};
      size_t new_size = 1;
      for (int i=0; i < new_rank; i++) {
        size_t const extent = checked_extent(bounds[i]);
        if constexpr (kokkos_debug) {
          if (extent != 0 && new_size > std::numeric_limits<size_t>::max()/extent) {
            Kokkos::abort("ERROR: Array_F::reshape dimension product overflow");
          }
        }
        new_size *= extent;
      }
      if (new_size != this_t::size()) {
        Kokkos::abort("ERROR: Resizing array with different total size");
      }
      auto loc = Array_F<new_kt,MemSpace>(this_t::data(),{newdims.l,newdims.u}...);
      KOKKOS_IF_ON_HOST(( loc.retain_reference_host(*this,0); ))
      return loc;
    }
    KOKKOS_INLINE_FUNCTION auto reshape(AB b1)                                           const { return reshape_all(b1); }
    KOKKOS_INLINE_FUNCTION auto reshape(AB b1,AB b2)                                     const { return reshape_all(b1,b2); }
    KOKKOS_INLINE_FUNCTION auto reshape(AB b1,AB b2,AB b3)                               const { return reshape_all(b1,b2,b3); }
    KOKKOS_INLINE_FUNCTION auto reshape(AB b1,AB b2,AB b3,AB b4)                         const { return reshape_all(b1,b2,b3,b4); }
    KOKKOS_INLINE_FUNCTION auto reshape(AB b1,AB b2,AB b3,AB b4,AB b5)                   const { return reshape_all(b1,b2,b3,b4,b5); }
    KOKKOS_INLINE_FUNCTION auto reshape(AB b1,AB b2,AB b3,AB b4,AB b5,AB b6)             const { return reshape_all(b1,b2,b3,b4,b5,b6); }
    KOKKOS_INLINE_FUNCTION auto reshape(AB b1,AB b2,AB b3,AB b4,AB b5,AB b6,AB b7)       const { return reshape_all(b1,b2,b3,b4,b5,b6,b7); }
    KOKKOS_INLINE_FUNCTION auto reshape(AB b1,AB b2,AB b3,AB b4,AB b5,AB b6,AB b7,AB b8) const { return reshape_all(b1,b2,b3,b4,b5,b6,b7,b8); }


    KOKKOS_INLINE_FUNCTION auto collapse(std::integral auto lb = 1) const {
      ptrdiff_t const lower = static_cast<ptrdiff_t>(lb);
      if constexpr (kokkos_debug) {
        if (!this_t::is_allocated()) Kokkos::abort("ERROR: collapsing an unallocated Array_F");
        if ((!std::is_signed_v<decltype(lb)> && static_cast<size_t>(lb) >
             static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())) ||
            this_t::size()-1 > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) ||
            lower > std::numeric_limits<ptrdiff_t>::max()-static_cast<ptrdiff_t>(this_t::size()-1)) {
          Kokkos::abort("ERROR: Array_F::collapse upper-bound overflow");
        }
      }
      auto loc = Array_F<typename base_t::value_type *,MemSpace>(
          this_t::data(),{lower,lower+static_cast<ptrdiff_t>(this_t::size()-1)});
      KOKKOS_IF_ON_HOST(( loc.retain_reference_host(*this,0); ))
      return loc;
    }


    KOKKOS_INLINE_FUNCTION auto flatten() const { return this_t::collapse(1); }


    KOKKOS_INLINE_FUNCTION auto flatten(std::integral auto lb) const { return this_t::collapse(lb); }


    template <class ViewType>
    void deep_copy_to(ViewType const & them) const {
      if (them.size() != this->size()) Kokkos::abort("ERROR: calling deep_copy_to between differently sized arrays");
      if constexpr (yakl_auto_profile) timer_start("yakl::Array_F::deep_copy_to");
      Kokkos::deep_copy(them,*this);
      if constexpr (yakl_auto_profile) timer_stop("yakl::Array_F::deep_copy_to");
      if constexpr (yakl_auto_fence) Kokkos::fence();
      if constexpr (std::is_same_v<typename ViewType::memory_space,Kokkos::HostSpace>) Kokkos::fence();
    }


    auto createDeviceObject() const { return clone_object<yakl::DeviceSpace>(); }


    auto createHostObject() const { return clone_object<Kokkos::HostSpace>(); }


    auto createDeviceCopy() const {
      if constexpr (kokkos_debug) {
        if (!this_t::is_allocated()) Kokkos::abort("ERROR: copying an unallocated Array_F to the device");
      }
      auto ret = createDeviceObject();
      if constexpr (yakl_auto_profile) timer_start("yakl::Array_F::CreateDeviceCopy deep_copy");
      Kokkos::deep_copy( ret , *this );
      if constexpr (yakl_auto_profile) timer_stop("yakl::Array_F::CreateDeviceCopy deep_copy");
      if constexpr (yakl_auto_fence) Kokkos::fence();
      return ret;
    }


    auto createHostCopy() const {
      if constexpr (kokkos_debug) {
        if (!this_t::is_allocated()) Kokkos::abort("ERROR: copying an unallocated Array_F to the host");
      }
      auto ret = createHostObject();
      if constexpr (yakl_auto_profile) timer_start("yakl::Array_F::CreateHostCopy deep_copy");
      Kokkos::deep_copy( ret , *this );
      if constexpr (yakl_auto_profile) timer_stop("yakl::Array_F::CreateHostCopy deep_copy");
      if constexpr (yakl_auto_fence) Kokkos::fence();
      Kokkos::fence();
      return ret;
    }


    template <class scalar_t> requires std::is_arithmetic_v<scalar_t>
    Array_F<typename ViewType<scalar_t,this_t::rank()>::type,MemSpace> as() const {
      if constexpr (kokkos_debug) {
        if (!this_t::is_allocated()) Kokkos::abort("ERROR: converting an unallocated Array_F");
      }
      auto func = [&] <std::size_t... Is> (std::index_sequence<Is...>) {
        return Array_F<typename ViewType<scalar_t,this_t::rank()>::type,MemSpace>( this->label() , this->extent(Is)... );
      };
      auto ret = func(std::make_index_sequence<this_t::rank()>{});
      YAKL_SCOPE( me , *this );
      if constexpr (yakl_auto_profile) timer_start("yakl::Array_F::as");
      Kokkos::parallel_for( "yakl_as_copy" ,
                            Kokkos::RangePolicy<typename base_t::execution_space,Kokkos::IndexType<size_t>>(0,this->size()) ,
                            KOKKOS_LAMBDA (size_t i) {
        ret.data()[i] = me.data()[i];
      });
      if constexpr (yakl_auto_profile) timer_stop("yakl::Array_F::as");
      if constexpr (yakl_auto_fence) Kokkos::fence();
      for (int i=0; i < this_t::rank(); i++) { ret.lb[i] = this->lb[i]; }
      return ret;
    }


    KOKKOS_INLINE_FUNCTION auto extents() const {
      SArray_F<size_t,Bnds{1,static_cast<int>(this_t::rank())}> ret;
      for (int i=1; i <= this_t::rank(); i++) { ret(i) = this_t::extent(i-1); }
      return ret;
    }


    KOKKOS_INLINE_FUNCTION auto ubounds() const {
      SArray_F<ptrdiff_t,Bnds{1,static_cast<int>(this_t::rank())}> ret;
      for (int i=1; i <= this_t::rank(); i++) { ret(i) = lb[i-1] + this_t::extent(i-1)-1; }
      return ret;
    }


    KOKKOS_INLINE_FUNCTION auto lbounds() const {
      SArray_F<ptrdiff_t,Bnds{1,static_cast<int>(this_t::rank())}> ret;
      for (int i=1; i <= this_t::rank(); i++) { ret(i) = lb[i-1]; }
      return ret;
    }


    KOKKOS_INLINE_FUNCTION base_t::value_type * begin() const { return this_t::data(); }
    KOKKOS_INLINE_FUNCTION base_t::value_type * end  () const { return this_t::data()+this_t::size(); }


    inline friend std::ostream &operator<<( std::ostream& os , Array_F const & v ) {
      auto loc = v.createHostCopy(); // cout,cerr is expensive, so just create a host copy
      os << "Array_F [" << loc.label() << "], Dimensions [";
      for (int i = 0; i < loc.rank(); i++) { os << loc.extent(i) << (i<loc.rank()-1 ? "," : ""); }
      os << "] = " << loc.size() << " Elements:  ";
      for (size_t i = 0; i < loc.size(); i++) { os << loc.data()[i] << (i<loc.size()-1 ? " , " : ""); }
      os << std::endl;
      return os;
    }


    KOKKOS_INLINE_FUNCTION auto unpack_global_index(size_t iglob) const {
      SArray_F<ptrdiff_t,Bnds{1,this_t::rank()}> ret;
      if constexpr (kokkos_bounds_debug) {
        if (iglob >= this_t::size()) Kokkos::abort("ERROR: Array_F::unpack_global_index index out of bounds");
      }
      for (int i=1; i <= this_t::rank(); i++) {
        ret(i) = static_cast<ptrdiff_t>((iglob / this_t::stride(i-1)) % this_t::extent(i-1)) + lb[i-1];
      }
      return ret;
    }


    base_t const & get_View() const { return static_cast<base_t const &>(*this); }
    base_t       & get_View()       { return static_cast<base_t       &>(*this); }
  };

}
