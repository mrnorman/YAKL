
#pragma once

namespace yakl {


  struct CStyle { static constexpr bool is_cstyle = true; };
  struct FStyle { static constexpr bool is_fstyle = true; };

  template <class Type> inline constexpr bool is_CStyle = requires { requires Type::is_cstyle; };
  template <class Type> inline constexpr bool is_FStyle = requires { requires Type::is_fstyle; };



  namespace parallel_for_detail {

    KOKKOS_INLINE_FUNCTION ptrdiff_t checked_loop_bound(std::integral auto value) {
      if constexpr (kokkos_debug) {
        if (!std::in_range<ptrdiff_t>(value)) Kokkos::abort("ERROR: loop bound is not representable by ptrdiff_t");
      }
      return static_cast<ptrdiff_t>(value);
    }

    KOKKOS_INLINE_FUNCTION ptrdiff_t checked_loop_stride(std::integral auto value) {
      ptrdiff_t const stride = checked_loop_bound(value);
      if constexpr (kokkos_debug) {
        if (stride < 1) Kokkos::abort("ERROR: non-positive strides not supported.");
      }
      return stride;
    }

    KOKKOS_INLINE_FUNCTION size_t loop_index_range(ptrdiff_t l, ptrdiff_t u, ptrdiff_t s) {
      if constexpr (kokkos_debug) {
        if (s <= 0 || u < l) Kokkos::abort("ERROR: requesting the range of an invalid LoopSpec");
      }
      using unsigned_bound_t = std::make_unsigned_t<ptrdiff_t>;
      auto const difference = static_cast<unsigned_bound_t>(u)-static_cast<unsigned_bound_t>(l);
      auto const quotient = difference/static_cast<unsigned_bound_t>(s);
      if constexpr (kokkos_debug) {
        if (quotient == std::numeric_limits<unsigned_bound_t>::max()) {
          Kokkos::abort("ERROR: LoopSpec range overflow");
        }
      }
      return static_cast<size_t>(quotient+1);
    }

  }



  class LoopSpec {
  public:
    bool      static constexpr is_cstyle = true;
    bool      static constexpr is_fstyle = false;
    ptrdiff_t static constexpr default_lbound = 0;
    ptrdiff_t l, u, s;
    KOKKOS_INLINE_FUNCTION LoopSpec() : l(-1),u(-1),s(-1) { }
    KOKKOS_INLINE_FUNCTION LoopSpec(std::integral auto u) : l(default_lbound),u(default_lbound-1),s(1) {
      if constexpr (kokkos_debug) {
        if ((std::is_signed_v<decltype(u)> && u < 0) ||
            static_cast<size_t>(u) > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())) {
          Kokkos::abort("ERROR: loop extent must be nonnegative and representable by ptrdiff_t");
        }
      }
      this->u = static_cast<ptrdiff_t>(u)-1+default_lbound;
    }
    KOKKOS_INLINE_FUNCTION LoopSpec(std::integral auto l, std::integral auto u) :
        l(parallel_for_detail::checked_loop_bound(l)),u(parallel_for_detail::checked_loop_bound(u)),s(1) {
      if constexpr (kokkos_debug) { if (this->u < this->l) Kokkos::abort("ERROR: cannot specify an upper bound < lower bound"); }
    }
    KOKKOS_INLINE_FUNCTION LoopSpec(std::integral auto l, std::integral auto u, std::integral auto s) :
        l(parallel_for_detail::checked_loop_bound(l)),u(parallel_for_detail::checked_loop_bound(u)),
        s(parallel_for_detail::checked_loop_stride(s)) {
      if constexpr (kokkos_debug) { if (this->u < this->l) Kokkos::abort("ERROR: cannot specify an upper bound < lower bound"); }
    }
    KOKKOS_INLINE_FUNCTION bool   valid      () const { return this->s > 0; }
    KOKKOS_INLINE_FUNCTION size_t index_range() const {
      return parallel_for_detail::loop_index_range(l,u,s);
    }
  };



  class LoopSpec_F {
  public:
    bool      static constexpr is_cstyle = false;
    bool      static constexpr is_fstyle = true;
    ptrdiff_t static constexpr default_lbound = 1;
    ptrdiff_t l, u, s;
    KOKKOS_INLINE_FUNCTION LoopSpec_F() : l(-1),u(-1),s(-1) { }
    KOKKOS_INLINE_FUNCTION LoopSpec_F(std::integral auto u) : l(default_lbound),u(default_lbound-1),s(1) {
      if constexpr (kokkos_debug) {
        if ((std::is_signed_v<decltype(u)> && u < 0) ||
            static_cast<size_t>(u) > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())) {
          Kokkos::abort("ERROR: loop extent must be nonnegative and representable by ptrdiff_t");
        }
      }
      this->u = static_cast<ptrdiff_t>(u)-1+default_lbound;
    }
    KOKKOS_INLINE_FUNCTION LoopSpec_F(std::integral auto l, std::integral auto u) :
        l(parallel_for_detail::checked_loop_bound(l)),u(parallel_for_detail::checked_loop_bound(u)),s(1) {
      if constexpr (kokkos_debug) { if (this->u < this->l) Kokkos::abort("ERROR: cannot specify an upper bound < lower bound"); }
    }
    KOKKOS_INLINE_FUNCTION LoopSpec_F(std::integral auto l, std::integral auto u, std::integral auto s) :
        l(parallel_for_detail::checked_loop_bound(l)),u(parallel_for_detail::checked_loop_bound(u)),
        s(parallel_for_detail::checked_loop_stride(s)) {
      if constexpr (kokkos_debug) { if (this->u < this->l) Kokkos::abort("ERROR: cannot specify an upper bound < lower bound"); }
    }
    KOKKOS_INLINE_FUNCTION bool   valid      () const { return this->s > 0; }
    KOKKOS_INLINE_FUNCTION size_t index_range() const {
      return parallel_for_detail::loop_index_range(l,u,s);
    }
  };



  template <int N, class Style=CStyle, bool Simple=false> class Bounds;


  template<int N, class Style> class Bounds<N,Style,true> {
    public:
    using unsigned_t = size_t;
    bool       static constexpr is_cstyle      = is_CStyle<Style>;
    bool       static constexpr is_fstyle      = is_FStyle<Style>;
    unsigned_t static constexpr default_lbound = is_cstyle ? 0 : 1;
    unsigned_t nIter;
    std::array<unsigned_t,N> offs;
    KOKKOS_INLINE_FUNCTION static unsigned_t checked_extent(std::integral auto size) {
      if constexpr (kokkos_debug) {
        if (!std::in_range<unsigned_t>(size)) Kokkos::abort("ERROR: Bounds dimensions cannot be negative or overflow size_t");
      }
      return static_cast<unsigned_t>(size);
    }
    KOKKOS_INLINE_FUNCTION Bounds( std::integral auto... sizes ) {
      static_assert(sizeof...(sizes)==N,"ERROR: Bounds class creation with wrong number of loop bounds");
      std::array<unsigned_t,N> dims = { checked_extent(sizes)... };
      nIter = 1;
      for (int i=0; i < N; i++) {
        if constexpr (kokkos_debug) {
          if (dims[i] != 0 && nIter > std::numeric_limits<unsigned_t>::max()/dims[i]) {
            Kokkos::abort("ERROR: Bounds iteration-count overflow");
          }
        }
        nIter *= dims[i];
        offs[i] = 1;
        for (int j=i+1; j < N; j++) {
          if constexpr (kokkos_debug) {
            if (dims[j] != 0 && offs[i] > std::numeric_limits<unsigned_t>::max()/dims[j]) {
              Kokkos::abort("ERROR: Bounds offset overflow");
            }
          }
          offs[i] *= dims[j];
        }
      }
    }
    KOKKOS_INLINE_FUNCTION void check_global_index(unsigned_t iglob) const {
      if constexpr (kokkos_bounds_debug) {
        if (iglob >= nIter) Kokkos::abort("ERROR: Bounds::unpack global index out of bounds");
      }
    }

    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , unsigned_t & i0 ) const requires (N==1) {
      check_global_index(iglob);
      i0 = iglob        ;                        i0 += default_lbound;
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , unsigned_t & i0 ,
                                                           unsigned_t & i1 ) const requires (N==2) {
      check_global_index(iglob);
      i0 = iglob/offs[0];  iglob -= offs[0]*i0;  i0 += default_lbound;
      i1 = iglob        ;                        i1 += default_lbound;
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , unsigned_t & i0 ,
                                                           unsigned_t & i1 ,
                                                           unsigned_t & i2 ) const requires (N==3) {
      check_global_index(iglob);
      i0 = iglob/offs[0];  iglob -= offs[0]*i0;  i0 += default_lbound;
      i1 = iglob/offs[1];  iglob -= offs[1]*i1;  i1 += default_lbound;
      i2 = iglob        ;                        i2 += default_lbound;
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , unsigned_t & i0 ,
                                                           unsigned_t & i1 ,
                                                           unsigned_t & i2 ,
                                                           unsigned_t & i3 ) const requires (N==4) {
      check_global_index(iglob);
      i0 = iglob/offs[0];  iglob -= offs[0]*i0;  i0 += default_lbound;
      i1 = iglob/offs[1];  iglob -= offs[1]*i1;  i1 += default_lbound;
      i2 = iglob/offs[2];  iglob -= offs[2]*i2;  i2 += default_lbound;
      i3 = iglob        ;                        i3 += default_lbound;
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , unsigned_t & i0 ,
                                                           unsigned_t & i1 ,
                                                           unsigned_t & i2 ,
                                                           unsigned_t & i3 ,
                                                           unsigned_t & i4) const requires (N==5) {
      check_global_index(iglob);
      i0 = iglob/offs[0];  iglob -= offs[0]*i0;  i0 += default_lbound;
      i1 = iglob/offs[1];  iglob -= offs[1]*i1;  i1 += default_lbound;
      i2 = iglob/offs[2];  iglob -= offs[2]*i2;  i2 += default_lbound;
      i3 = iglob/offs[3];  iglob -= offs[3]*i3;  i3 += default_lbound;
      i4 = iglob        ;                        i4 += default_lbound;
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , unsigned_t & i0 ,
                                                           unsigned_t & i1 ,
                                                           unsigned_t & i2 ,
                                                           unsigned_t & i3 ,
                                                           unsigned_t & i4 ,
                                                           unsigned_t & i5) const requires (N==6) {
      check_global_index(iglob);
      i0 = iglob/offs[0];  iglob -= offs[0]*i0;  i0 += default_lbound;
      i1 = iglob/offs[1];  iglob -= offs[1]*i1;  i1 += default_lbound;
      i2 = iglob/offs[2];  iglob -= offs[2]*i2;  i2 += default_lbound;
      i3 = iglob/offs[3];  iglob -= offs[3]*i3;  i3 += default_lbound;
      i4 = iglob/offs[4];  iglob -= offs[4]*i4;  i4 += default_lbound;
      i5 = iglob        ;                        i5 += default_lbound;
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , unsigned_t & i0 ,
                                                           unsigned_t & i1 ,
                                                           unsigned_t & i2 ,
                                                           unsigned_t & i3 ,
                                                           unsigned_t & i4 ,
                                                           unsigned_t & i5 ,
                                                           unsigned_t & i6) const requires (N==7) {
      check_global_index(iglob);
      i0 = iglob/offs[0];  iglob -= offs[0]*i0;  i0 += default_lbound;
      i1 = iglob/offs[1];  iglob -= offs[1]*i1;  i1 += default_lbound;
      i2 = iglob/offs[2];  iglob -= offs[2]*i2;  i2 += default_lbound;
      i3 = iglob/offs[3];  iglob -= offs[3]*i3;  i3 += default_lbound;
      i4 = iglob/offs[4];  iglob -= offs[4]*i4;  i4 += default_lbound;
      i5 = iglob/offs[5];  iglob -= offs[5]*i5;  i5 += default_lbound;
      i6 = iglob        ;                        i6 += default_lbound;
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , unsigned_t & i0 ,
                                                           unsigned_t & i1 ,
                                                           unsigned_t & i2 ,
                                                           unsigned_t & i3 ,
                                                           unsigned_t & i4 ,
                                                           unsigned_t & i5 ,
                                                           unsigned_t & i6 ,
                                                           unsigned_t & i7) const requires (N==8) {
      check_global_index(iglob);
      i0 = iglob/offs[0];  iglob -= offs[0]*i0;  i0 += default_lbound;
      i1 = iglob/offs[1];  iglob -= offs[1]*i1;  i1 += default_lbound;
      i2 = iglob/offs[2];  iglob -= offs[2]*i2;  i2 += default_lbound;
      i3 = iglob/offs[3];  iglob -= offs[3]*i3;  i3 += default_lbound;
      i4 = iglob/offs[4];  iglob -= offs[4]*i4;  i4 += default_lbound;
      i5 = iglob/offs[5];  iglob -= offs[5]*i5;  i5 += default_lbound;
      i6 = iglob/offs[6];  iglob -= offs[6]*i6;  i6 += default_lbound;
      i7 = iglob        ;                        i7 += default_lbound;
    }
  };


  template<int N, class Style> class Bounds<N,Style,false> {
    public:
    using unsigned_t = size_t;
    using signed_t   = ptrdiff_t;
    bool   static constexpr is_cstyle = is_CStyle<Style>;
    bool   static constexpr is_fstyle = is_FStyle<Style>;
    using LS = std::conditional_t<is_cstyle,LoopSpec,LoopSpec_F>;
    unsigned_t nIter;

    std::array<unsigned_t,N> offs;
    std::array<signed_t  ,N> lbounds;
    std::array<unsigned_t,N> strides;

    template <class... BNDS> requires (std::is_same_v<BNDS,LS> && ...)
    KOKKOS_INLINE_FUNCTION void init( BNDS... bnds ) {
      static_assert(sizeof...(bnds) == N,"Error: Bounds::init called with wrong number of bounds parameters");
      if constexpr (kokkos_debug) {
        if (((!bnds.valid() || bnds.u < bnds.l) || ...)) {
          Kokkos::abort("ERROR: Bounds created from an invalid LoopSpec");
        }
      }
      std::array<unsigned_t,N> dims = { bnds.index_range()... };
      lbounds                       = { static_cast<signed_t  >(bnds.l)... };
      strides                       = { static_cast<unsigned_t>(bnds.s)... };
      nIter = 1;
      for (int i=0; i < N; i++) {
        if constexpr (kokkos_debug) {
          if (dims[i] != 0 && nIter > std::numeric_limits<unsigned_t>::max()/dims[i]) {
            Kokkos::abort("ERROR: Bounds iteration-count overflow");
          }
        }
        nIter *= dims[i];
        offs[i] = 1;
        for (int j=i+1; j < N; j++) {
          if constexpr (kokkos_debug) {
            if (dims[j] != 0 && offs[i] > std::numeric_limits<unsigned_t>::max()/dims[j]) {
              Kokkos::abort("ERROR: Bounds offset overflow");
            }
          }
          offs[i] *= dims[j];
        }
      }
    }

    KOKKOS_INLINE_FUNCTION void check_global_index(unsigned_t iglob) const {
      if constexpr (kokkos_bounds_debug) {
        if (iglob >= nIter) Kokkos::abort("ERROR: Bounds::unpack global index out of bounds");
      }
    }

    KOKKOS_INLINE_FUNCTION signed_t index_from_coordinate(unsigned_t coordinate, int dim) const {
      if constexpr (kokkos_debug) {
        if (coordinate != 0 && strides[dim] > std::numeric_limits<unsigned_t>::max()/coordinate) {
          Kokkos::abort("ERROR: Bounds index reconstruction overflow");
        }
      }
      unsigned_t const delta       = coordinate*strides[dim];
      unsigned_t constexpr max_pos = static_cast<unsigned_t>(std::numeric_limits<signed_t>::max());
      signed_t const lower         = lbounds[dim];
      if (lower >= 0) {
        if constexpr (kokkos_debug) {
          if (delta > max_pos-static_cast<unsigned_t>(lower)) {
            Kokkos::abort("ERROR: Bounds index reconstruction overflow");
          }
        }
        return lower+static_cast<signed_t>(delta);
      }
      unsigned_t const lower_magnitude = static_cast<unsigned_t>(-(lower+1))+1;
      if (delta < lower_magnitude) {
        unsigned_t const result_magnitude = lower_magnitude-delta;
        if (result_magnitude == max_pos+1) return std::numeric_limits<signed_t>::min();
        return -static_cast<signed_t>(result_magnitude);
      }
      unsigned_t const result = delta-lower_magnitude;
      if constexpr (kokkos_debug) {
        if (result > max_pos) Kokkos::abort("ERROR: Bounds index reconstruction overflow");
      }
      return static_cast<signed_t>(result);
    }

    KOKKOS_INLINE_FUNCTION signed_t unpack_dimension(unsigned_t & iglob, int dim) const {
      unsigned_t const coordinate = dim == N-1 ? iglob : iglob/offs[dim];
      if (dim != N-1) iglob -= offs[dim]*coordinate;
      return index_from_coordinate(coordinate,dim);
    }

    KOKKOS_INLINE_FUNCTION Bounds(LS s0) {
      static_assert(N==1,"ERROR: Creating Bounds with wrong number of bounds parameters");
      init(s0);
    }
    KOKKOS_INLINE_FUNCTION Bounds(LS s0,LS s1) {
      static_assert(N==2,"ERROR: Creating Bounds with wrong number of bounds parameters");
      init(s0,s1);
    }
    KOKKOS_INLINE_FUNCTION Bounds(LS s0,LS s1,LS s2) {
      static_assert(N==3,"ERROR: Creating Bounds with wrong number of bounds parameters");
      init(s0,s1,s2);
    }
    KOKKOS_INLINE_FUNCTION Bounds(LS s0,LS s1,LS s2,LS s3) {
      static_assert(N==4,"ERROR: Creating Bounds with wrong number of bounds parameters");
      init(s0,s1,s2,s3);
    }
    KOKKOS_INLINE_FUNCTION Bounds(LS s0,LS s1,LS s2,LS s3,LS s4) {
      static_assert(N==5,"ERROR: Creating Bounds with wrong number of bounds parameters");
      init(s0,s1,s2,s3,s4);
    }
    KOKKOS_INLINE_FUNCTION Bounds(LS s0,LS s1,LS s2,LS s3,LS s4,LS s5) {
      static_assert(N==6,"ERROR: Creating Bounds with wrong number of bounds parameters");
      init(s0,s1,s2,s3,s4,s5);
    }
    KOKKOS_INLINE_FUNCTION Bounds(LS s0,LS s1,LS s2,LS s3,LS s4,LS s5,LS s6) {
      static_assert(N==7,"ERROR: Creating Bounds with wrong number of bounds parameters");
      init(s0,s1,s2,s3,s4,s5,s6);
    }
    KOKKOS_INLINE_FUNCTION Bounds(LS s0,LS s1,LS s2,LS s3,LS s4,LS s5,LS s6,LS s7) {
      static_assert(N==8,"ERROR: Creating Bounds with wrong number of bounds parameters");
      init(s0,s1,s2,s3,s4,s5,s6,s7);
    }

    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , signed_t & i0 ) const requires (N==1) {
      check_global_index(iglob);
      i0 = unpack_dimension(iglob,0);
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , signed_t & i0 ,
                                                           signed_t & i1 ) const requires (N==2) {
      check_global_index(iglob);
      i0 = unpack_dimension(iglob,0);
      i1 = unpack_dimension(iglob,1);
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , signed_t & i0 ,
                                                           signed_t & i1 ,
                                                           signed_t & i2 ) const requires (N==3) {
      check_global_index(iglob);
      i0 = unpack_dimension(iglob,0);
      i1 = unpack_dimension(iglob,1);
      i2 = unpack_dimension(iglob,2);
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , signed_t & i0 ,
                                                           signed_t & i1 ,
                                                           signed_t & i2 ,
                                                           signed_t & i3 ) const requires (N==4) {
      check_global_index(iglob);
      i0 = unpack_dimension(iglob,0);
      i1 = unpack_dimension(iglob,1);
      i2 = unpack_dimension(iglob,2);
      i3 = unpack_dimension(iglob,3);
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , signed_t & i0 ,
                                                           signed_t & i1 ,
                                                           signed_t & i2 ,
                                                           signed_t & i3 ,
                                                           signed_t & i4) const requires (N==5) {
      check_global_index(iglob);
      i0 = unpack_dimension(iglob,0);
      i1 = unpack_dimension(iglob,1);
      i2 = unpack_dimension(iglob,2);
      i3 = unpack_dimension(iglob,3);
      i4 = unpack_dimension(iglob,4);
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , signed_t & i0 ,
                                                           signed_t & i1 ,
                                                           signed_t & i2 ,
                                                           signed_t & i3 ,
                                                           signed_t & i4 ,
                                                           signed_t & i5) const requires (N==6) {
      check_global_index(iglob);
      i0 = unpack_dimension(iglob,0);
      i1 = unpack_dimension(iglob,1);
      i2 = unpack_dimension(iglob,2);
      i3 = unpack_dimension(iglob,3);
      i4 = unpack_dimension(iglob,4);
      i5 = unpack_dimension(iglob,5);
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , signed_t & i0 ,
                                                           signed_t & i1 ,
                                                           signed_t & i2 ,
                                                           signed_t & i3 ,
                                                           signed_t & i4 ,
                                                           signed_t & i5 ,
                                                           signed_t & i6) const requires (N==7) {
      check_global_index(iglob);
      i0 = unpack_dimension(iglob,0);
      i1 = unpack_dimension(iglob,1);
      i2 = unpack_dimension(iglob,2);
      i3 = unpack_dimension(iglob,3);
      i4 = unpack_dimension(iglob,4);
      i5 = unpack_dimension(iglob,5);
      i6 = unpack_dimension(iglob,6);
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , signed_t & i0 ,
                                                           signed_t & i1 ,
                                                           signed_t & i2 ,
                                                           signed_t & i3 ,
                                                           signed_t & i4 ,
                                                           signed_t & i5 ,
                                                           signed_t & i6 ,
                                                           signed_t & i7) const requires (N==8) {
      check_global_index(iglob);
      i0 = unpack_dimension(iglob,0);
      i1 = unpack_dimension(iglob,1);
      i2 = unpack_dimension(iglob,2);
      i3 = unpack_dimension(iglob,3);
      i4 = unpack_dimension(iglob,4);
      i5 = unpack_dimension(iglob,5);
      i6 = unpack_dimension(iglob,6);
      i7 = unpack_dimension(iglob,7);
    }
  };



  template <int N> using SimpleBounds   = Bounds<N,CStyle,true >;
  template <int N> using SimpleBounds_F = Bounds<N,FStyle,true >;
  template <int N> using Bounds_F       = Bounds<N,FStyle,false>;



  template <int MaxThreadsPerBlock=0> requires (MaxThreadsPerBlock >= 0) struct Config {
    int static constexpr Thr = MaxThreadsPerBlock;
    size_t tile;

    KOKKOS_INLINE_FUNCTION Config() : tile(1) {}

    template <std::integral T>
    KOKKOS_INLINE_FUNCTION explicit Config(T tile) : tile(static_cast<size_t>(tile)) {
      if constexpr (kokkos_debug) {
        if (!std::in_range<size_t>(tile) || tile == 0) Kokkos::abort("ERROR: Config tile size must be positive");
      }
    }
  };



  template <class F, int N, bool simple, class Style>
  KOKKOS_FORCEINLINE_FUNCTION void call_parallel_for_functor( Bounds<N,Style,simple> const & bounds ,
                                                               F                      const & f      ,
                                                               size_t                         iglob  ) {
    if constexpr (simple) {
      if constexpr (N==1) {
        size_t i0; bounds.unpack(iglob,i0); f(i0);
      } else if constexpr (N==2) {
        size_t i0,i1; bounds.unpack(iglob,i0,i1); f(i0,i1);
      } else if constexpr (N==3) {
        size_t i0,i1,i2; bounds.unpack(iglob,i0,i1,i2); f(i0,i1,i2);
      } else if constexpr (N==4) {
        size_t i0,i1,i2,i3; bounds.unpack(iglob,i0,i1,i2,i3); f(i0,i1,i2,i3);
      } else if constexpr (N==5) {
        size_t i0,i1,i2,i3,i4; bounds.unpack(iglob,i0,i1,i2,i3,i4); f(i0,i1,i2,i3,i4);
      } else if constexpr (N==6) {
        size_t i0,i1,i2,i3,i4,i5; bounds.unpack(iglob,i0,i1,i2,i3,i4,i5); f(i0,i1,i2,i3,i4,i5);
      } else if constexpr (N==7) {
        size_t i0,i1,i2,i3,i4,i5,i6; bounds.unpack(iglob,i0,i1,i2,i3,i4,i5,i6); f(i0,i1,i2,i3,i4,i5,i6);
      } else if constexpr (N==8) {
        size_t i0,i1,i2,i3,i4,i5,i6,i7; bounds.unpack(iglob,i0,i1,i2,i3,i4,i5,i6,i7); f(i0,i1,i2,i3,i4,i5,i6,i7);
      }
    } else {
      if constexpr (N==1) {
        ptrdiff_t i0; bounds.unpack(iglob,i0); f(i0);
      } else if constexpr (N==2) {
        ptrdiff_t i0,i1; bounds.unpack(iglob,i0,i1); f(i0,i1);
      } else if constexpr (N==3) {
        ptrdiff_t i0,i1,i2; bounds.unpack(iglob,i0,i1,i2); f(i0,i1,i2);
      } else if constexpr (N==4) {
        ptrdiff_t i0,i1,i2,i3; bounds.unpack(iglob,i0,i1,i2,i3); f(i0,i1,i2,i3);
      } else if constexpr (N==5) {
        ptrdiff_t i0,i1,i2,i3,i4; bounds.unpack(iglob,i0,i1,i2,i3,i4); f(i0,i1,i2,i3,i4);
      } else if constexpr (N==6) {
        ptrdiff_t i0,i1,i2,i3,i4,i5; bounds.unpack(iglob,i0,i1,i2,i3,i4,i5); f(i0,i1,i2,i3,i4,i5);
      } else if constexpr (N==7) {
        ptrdiff_t i0,i1,i2,i3,i4,i5,i6; bounds.unpack(iglob,i0,i1,i2,i3,i4,i5,i6); f(i0,i1,i2,i3,i4,i5,i6);
      } else if constexpr (N==8) {
        ptrdiff_t i0,i1,i2,i3,i4,i5,i6,i7; bounds.unpack(iglob,i0,i1,i2,i3,i4,i5,i6,i7); f(i0,i1,i2,i3,i4,i5,i6,i7);
      }
    }
  }



  template <int MaxThreadsPerBlock, class F, int N, bool simple, class Style>
  inline void launch_parallel_for_untiled( std::string                    str    ,
                                           Bounds<N,Style,simple> const & bounds ,
                                           F                      const & f      ) {
    using Policy = Kokkos::RangePolicy<Kokkos::LaunchBounds<MaxThreadsPerBlock,0>,Kokkos::IndexType<size_t>>;
    Kokkos::parallel_for( str , Policy(0,bounds.nIter) , KOKKOS_LAMBDA (size_t iglob) {
      call_parallel_for_functor(bounds,f,iglob);
    });
  }



  template <int MaxThreadsPerBlock, class F, int N, bool simple, class Style>
  inline void launch_parallel_for_tiled( std::string                    str    ,
                                         Bounds<N,Style,simple> const & bounds ,
                                         F                      const & f      ,
                                         size_t                         tile   ) {
    std::array<size_t,N> boundDims;
    std::array<size_t,N> tileOffs;
    boundDims[0] = bounds.nIter/bounds.offs[0];
    for (int d=1; d < N; d++) boundDims[d] = bounds.offs[d-1]/bounds.offs[d];
    size_t nTiles = 1;
    for (int d=0; d < N; d++) {
      size_t const tileCount = (boundDims[d]-1)/tile+1;
      if constexpr (kokkos_debug) {
        if (tileCount != 0 && nTiles > std::numeric_limits<size_t>::max()/tileCount) {
          Kokkos::abort("ERROR: tiled parallel_for iteration-count overflow");
        }
      }
      nTiles *= tileCount;
      tileOffs[d] = 1;
      for (int j=d+1; j < N; j++) tileOffs[d] *= (boundDims[j]-1)/tile+1;
    }

    using Policy = Kokkos::RangePolicy<Kokkos::LaunchBounds<MaxThreadsPerBlock,0>,Kokkos::IndexType<size_t>>;
    Kokkos::parallel_for( str , Policy(0,nTiles) , KOKKOS_LAMBDA (size_t tileIndex) {
      std::array<size_t,N> starts;
      std::array<size_t,N> localDims;
      size_t remainder = tileIndex;
      size_t localIterations = 1;
      for (int d=0; d < N; d++) {
        size_t const tileCoord = remainder/tileOffs[d];
        remainder -= tileCoord*tileOffs[d];
        starts[d] = tileCoord*tile;
        size_t const remaining = boundDims[d]-starts[d];
        localDims[d] = remaining < tile ? remaining : tile;
        localIterations *= localDims[d];
      }
      for (size_t localIndex=0; localIndex < localIterations; localIndex++) {
        remainder = localIndex;
        size_t iglob = 0;
        for (int d=N-1; d >= 0; d--) {
          size_t const coord = remainder%localDims[d];
          remainder /= localDims[d];
          iglob += (starts[d]+coord)*bounds.offs[d];
        }
        call_parallel_for_functor(bounds,f,iglob);
      }
    });
  }



  template <class F, int N, bool simple, class Style>
  inline void parallel_for( std::string                    str    ,
                            Bounds<N,Style,simple> const & bounds ,
                            F                      const & f      ) {
    if (bounds.nIter == 0) return;
    if constexpr (yakl_auto_profile) timer_start(str);
    launch_parallel_for_untiled<0>(str,bounds,f);
    if constexpr (yakl_auto_profile) timer_stop(str);
    if constexpr (yakl_auto_fence) Kokkos::fence();
  }



  template <class F, int N, bool simple, class Style, int MaxThreadsPerBlock>
  inline void parallel_for( std::string                    str    ,
                            Bounds<N,Style,simple> const & bounds ,
                            F                      const & f      ,
                            Config<MaxThreadsPerBlock>             config ) {
    if (bounds.nIter == 0) return;
    if constexpr (kokkos_debug) {
      if (config.tile == 0) Kokkos::abort("ERROR: Config tile size must be positive");
    }
    if constexpr (yakl_auto_profile) timer_start(str);
    if (config.tile == 1) launch_parallel_for_untiled<MaxThreadsPerBlock>(str,bounds,f);
    else                  launch_parallel_for_tiled  <MaxThreadsPerBlock>(str,bounds,f,config.tile);
    if constexpr (yakl_auto_profile) timer_stop(str);
    if constexpr (yakl_auto_fence) Kokkos::fence();
  }



  template <class F, int N, bool simple, class Style>
  inline void parallel_for( Bounds<N,Style,simple> const & bounds , F const & f ) {
    parallel_for( YAKL_AUTO_LABEL() , bounds , f );
  }

  template <class F, int N, bool simple, class Style, int MaxThreadsPerBlock>
  inline void parallel_for( Bounds<N,Style,simple> const & bounds , F const & f , Config<MaxThreadsPerBlock> config ) {
    parallel_for( YAKL_AUTO_LABEL() , bounds , f , config );
  }

  template <class F>
  inline void parallel_for( std::integral auto bnd , F const & f ) {
    parallel_for( YAKL_AUTO_LABEL() , Bounds<1,CStyle,true>(bnd) , f );
  }

  template <class F, int MaxThreadsPerBlock>
  inline void parallel_for( std::integral auto bnd , F const & f , Config<MaxThreadsPerBlock> config ) {
    parallel_for( YAKL_AUTO_LABEL() , Bounds<1,CStyle,true>(bnd) , f , config );
  }

  template <class F>
  inline void parallel_for( std::string str , std::integral auto bnd , F const & f ) {
    parallel_for( str , Bounds<1,CStyle,true>(bnd) , f );
  }

  template <class F, int MaxThreadsPerBlock>
  inline void parallel_for( std::string str , std::integral auto bnd , F const & f , Config<MaxThreadsPerBlock> config ) {
    parallel_for( str , Bounds<1,CStyle,true>(bnd) , f , config );
  }



  template <class F, int N, bool simple>
  inline void parallel_for_F( std::string str , Bounds<N,FStyle,simple> const & bounds , F const & f ) {
    parallel_for<F,N,simple,FStyle>( str , bounds , f );
  }

  template <class F, int N, bool simple, int MaxThreadsPerBlock>
  inline void parallel_for_F( std::string str , Bounds<N,FStyle,simple> const & bounds , F const & f ,
                              Config<MaxThreadsPerBlock> config ) {
    parallel_for<F,N,simple,FStyle>( str , bounds , f , config );
  }

  template <class F, int N, bool simple>
  inline void parallel_for_F( Bounds<N,FStyle,simple> const & bounds , F const & f ) {
    parallel_for<F,N,simple,FStyle>( YAKL_AUTO_LABEL() , bounds , f );
  }

  template <class F, int N, bool simple, int MaxThreadsPerBlock>
  inline void parallel_for_F( Bounds<N,FStyle,simple> const & bounds , F const & f , Config<MaxThreadsPerBlock> config ) {
    parallel_for<F,N,simple,FStyle>( YAKL_AUTO_LABEL() , bounds , f , config );
  }

  template <class F>
  inline void parallel_for_F( std::integral auto bnd , F const & f ) {
    parallel_for<F,1,true,FStyle>( YAKL_AUTO_LABEL() , Bounds<1,FStyle,true>(bnd) , f );
  }

  template <class F, int MaxThreadsPerBlock>
  inline void parallel_for_F( std::integral auto bnd , F const & f , Config<MaxThreadsPerBlock> config ) {
    parallel_for<F,1,true,FStyle>( YAKL_AUTO_LABEL() , Bounds<1,FStyle,true>(bnd) , f , config );
  }

  template <class F>
  inline void parallel_for_F( std::string str , std::integral auto bnd , F const & f ) {
    parallel_for<F,1,true,FStyle>( str , Bounds<1,FStyle,true>(bnd) , f );
  }

  template <class F, int MaxThreadsPerBlock>
  inline void parallel_for_F( std::string str , std::integral auto bnd , F const & f , Config<MaxThreadsPerBlock> config ) {
    parallel_for<F,1,true,FStyle>( str , Bounds<1,FStyle,true>(bnd) , f , config );
  }

}
