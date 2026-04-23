
#pragma once

namespace yakl {


  struct CStyle { static constexpr bool is_cstyle = true; };
  struct FStyle { static constexpr bool is_fstyle = true; };

  template <class Type> inline constexpr bool is_CStyle = requires { requires Type::is_cstyle; };
  template <class Type> inline constexpr bool is_FStyle = requires { requires Type::is_fstyle; };



  template <class Style = CStyle> class LoopSpec {
  public:
    bool      static constexpr is_cstyle = is_CStyle<Style>;
    bool      static constexpr is_fstyle = is_FStyle<Style>;
    ptrdiff_t static constexpr default_lbound = is_cstyle ? 0 : 1;
    ptrdiff_t l, u, s;
    KOKKOS_INLINE_FUNCTION LoopSpec() : l(-1),u(-1),s(-1) { }
    KOKKOS_INLINE_FUNCTION LoopSpec(std::integral auto u) : l(default_lbound),u(u-1+default_lbound),s(1) { }
    KOKKOS_INLINE_FUNCTION LoopSpec(std::integral auto l, std::integral auto u) : l(l),u(u),s(1) {
      if constexpr (kokkos_debug) { if (u < l) Kokkos::abort("ERROR: cannot specify an upper bound < lower bound"); }
    }
    KOKKOS_INLINE_FUNCTION LoopSpec(std::integral auto l, std::integral auto u, std::integral auto s) : l(l),u(u),s(s) {
      if constexpr (kokkos_debug) { if (u < l) Kokkos::abort("ERROR: cannot specify an upper bound < lower bound"); }
      if constexpr (kokkos_debug) { if (s < 1) Kokkos::abort("ERROR: non-positive strides not supported."); }
    }
    KOKKOS_INLINE_FUNCTION bool   valid      () const { return this->s > 0; }
    KOKKOS_INLINE_FUNCTION size_t index_range() const { return this->u-this->l+1; }
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
    KOKKOS_INLINE_FUNCTION Bounds( std::integral auto... sizes ) {
      static_assert(sizeof...(sizes)==N,"ERROR: Bounds class creation with wrong number of loop bounds");
      std::array<unsigned_t,N> dims = { static_cast<unsigned_t>(sizes)... };
      nIter = 1;
      for (int i=0; i < N; i++) {
        nIter *= dims[i];
        offs[i] = 1;
        for (int j=i+1; j < N; j++) { offs[i] *= dims[j]; }
      }
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , unsigned_t & i0 ) const requires (N==1) {
      i0 = iglob        ;                        i0 += default_lbound;
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , unsigned_t & i0 ,
                                                           unsigned_t & i1 ) const requires (N==2) {
      i0 = iglob/offs[0];  iglob -= offs[0]*i0;  i0 += default_lbound;
      i1 = iglob        ;                        i1 += default_lbound;
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , unsigned_t & i0 ,
                                                           unsigned_t & i1 ,
                                                           unsigned_t & i2 ) const requires (N==3) {
      i0 = iglob/offs[0];  iglob -= offs[0]*i0;  i0 += default_lbound;
      i1 = iglob/offs[1];  iglob -= offs[1]*i1;  i1 += default_lbound;
      i2 = iglob        ;                        i2 += default_lbound;
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , unsigned_t & i0 ,
                                                           unsigned_t & i1 ,
                                                           unsigned_t & i2 ,
                                                           unsigned_t & i3 ) const requires (N==4) {
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
    unsigned_t nIter;

    std::array<unsigned_t,N> offs;
    std::array<signed_t  ,N> lbounds;
    std::array<unsigned_t,N> strides;

    template <class... BNDS> requires (std::is_same_v<BNDS,LoopSpec<Style>> && ...)
    KOKKOS_INLINE_FUNCTION void init( BNDS... bnds ) {
      static_assert(sizeof...(bnds) == N,"Error: Bounds::init called with wrong number of bounds parameters");
      std::array<unsigned_t,N> dims = { static_cast<unsigned_t>((bnds.u-bnds.l+1)/bnds.s)... };
      lbounds                       = { static_cast<signed_t  >(bnds.l)... };
      strides                       = { static_cast<unsigned_t>(bnds.s)... };
      nIter = 1;
      for (int i=0; i < N; i++) {
        nIter *= dims[i];
        offs[i] = 1;
        for (int j=i+1; j < N; j++) { offs[i] *= dims[j]; }
      }
    }

    using LS = LoopSpec<Style>;
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
      i0 = iglob        ;                        i0 = i0*strides[0]+lbounds[0];
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , signed_t & i0 ,
                                                           signed_t & i1 ) const requires (N==2) {
      i0 = iglob/offs[0];  iglob -= offs[0]*i0;  i0 = i0*strides[0]+lbounds[0];
      i1 = iglob        ;                        i1 = i1*strides[1]+lbounds[1];
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , signed_t & i0 ,
                                                           signed_t & i1 ,
                                                           signed_t & i2 ) const requires (N==3) {
      i0 = iglob/offs[0];  iglob -= offs[0]*i0;  i0 = i0*strides[0]+lbounds[0];
      i1 = iglob/offs[1];  iglob -= offs[1]*i1;  i1 = i1*strides[1]+lbounds[1];
      i2 = iglob        ;                        i2 = i2*strides[2]+lbounds[2];
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , signed_t & i0 ,
                                                           signed_t & i1 ,
                                                           signed_t & i2 ,
                                                           signed_t & i3 ) const requires (N==4) {
      i0 = iglob/offs[0];  iglob -= offs[0]*i0;  i0 = i0*strides[0]+lbounds[0];
      i1 = iglob/offs[1];  iglob -= offs[1]*i1;  i1 = i1*strides[1]+lbounds[1];
      i2 = iglob/offs[2];  iglob -= offs[2]*i2;  i2 = i2*strides[2]+lbounds[2];
      i3 = iglob        ;                        i3 = i3*strides[3]+lbounds[3];
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , signed_t & i0 ,
                                                           signed_t & i1 ,
                                                           signed_t & i2 ,
                                                           signed_t & i3 ,
                                                           signed_t & i4) const requires (N==5) {
      i0 = iglob/offs[0];  iglob -= offs[0]*i0;  i0 = i0*strides[0]+lbounds[0];
      i1 = iglob/offs[1];  iglob -= offs[1]*i1;  i1 = i1*strides[1]+lbounds[1];
      i2 = iglob/offs[2];  iglob -= offs[2]*i2;  i2 = i2*strides[2]+lbounds[2];
      i3 = iglob/offs[3];  iglob -= offs[3]*i3;  i3 = i3*strides[3]+lbounds[3];
      i4 = iglob        ;                        i4 = i4*strides[4]+lbounds[4];
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , signed_t & i0 ,
                                                           signed_t & i1 ,
                                                           signed_t & i2 ,
                                                           signed_t & i3 ,
                                                           signed_t & i4 ,
                                                           signed_t & i5) const requires (N==6) {
      i0 = iglob/offs[0];  iglob -= offs[0]*i0;  i0 = i0*strides[0]+lbounds[0];
      i1 = iglob/offs[1];  iglob -= offs[1]*i1;  i1 = i1*strides[1]+lbounds[1];
      i2 = iglob/offs[2];  iglob -= offs[2]*i2;  i2 = i2*strides[2]+lbounds[2];
      i3 = iglob/offs[3];  iglob -= offs[3]*i3;  i3 = i3*strides[3]+lbounds[3];
      i4 = iglob/offs[4];  iglob -= offs[4]*i4;  i4 = i4*strides[4]+lbounds[4];
      i5 = iglob        ;                        i5 = i5*strides[5]+lbounds[5];
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , signed_t & i0 ,
                                                           signed_t & i1 ,
                                                           signed_t & i2 ,
                                                           signed_t & i3 ,
                                                           signed_t & i4 ,
                                                           signed_t & i5 ,
                                                           signed_t & i6) const requires (N==7) {
      i0 = iglob/offs[0];  iglob -= offs[0]*i0;  i0 = i0*strides[0]+lbounds[0];
      i1 = iglob/offs[1];  iglob -= offs[1]*i1;  i1 = i1*strides[1]+lbounds[1];
      i2 = iglob/offs[2];  iglob -= offs[2]*i2;  i2 = i2*strides[2]+lbounds[2];
      i3 = iglob/offs[3];  iglob -= offs[3]*i3;  i3 = i3*strides[3]+lbounds[3];
      i4 = iglob/offs[4];  iglob -= offs[4]*i4;  i4 = i4*strides[4]+lbounds[4];
      i5 = iglob/offs[5];  iglob -= offs[5]*i5;  i5 = i5*strides[5]+lbounds[5];
      i6 = iglob        ;                        i6 = i6*strides[6]+lbounds[6];
    }
    KOKKOS_INLINE_FUNCTION void unpack( unsigned_t iglob , signed_t & i0 ,
                                                           signed_t & i1 ,
                                                           signed_t & i2 ,
                                                           signed_t & i3 ,
                                                           signed_t & i4 ,
                                                           signed_t & i5 ,
                                                           signed_t & i6 ,
                                                           signed_t & i7) const requires (N==8) {
      i0 = iglob/offs[0];  iglob -= offs[0]*i0;  i0 = i0*strides[0]+lbounds[0];
      i1 = iglob/offs[1];  iglob -= offs[1]*i1;  i1 = i1*strides[1]+lbounds[1];
      i2 = iglob/offs[2];  iglob -= offs[2]*i2;  i2 = i2*strides[2]+lbounds[2];
      i3 = iglob/offs[3];  iglob -= offs[3]*i3;  i3 = i3*strides[3]+lbounds[3];
      i4 = iglob/offs[4];  iglob -= offs[4]*i4;  i4 = i4*strides[4]+lbounds[4];
      i5 = iglob/offs[5];  iglob -= offs[5]*i5;  i5 = i5*strides[5]+lbounds[5];
      i6 = iglob/offs[6];  iglob -= offs[6]*i6;  i6 = i6*strides[6]+lbounds[6];
      i7 = iglob        ;                        i7 = i7*strides[7]+lbounds[7];
    }
  };



  template <int N> using SimpleBounds   = Bounds<N,CStyle,true >;
  template <int N> using SimpleBounds_F = Bounds<N,FStyle,true >;
  template <int N> using Bounds_F       = Bounds<N,FStyle,false>;



  template <int MaxThreadsPerBlock, size_t Strip> struct Config {
    int static constexpr Thr = MaxThreadsPerBlock;
    int static constexpr Str = Strip;
  };



  template <class F, int N, bool simple, class Style, int MaxThreadsPerBlock=0, size_t Strip=1>
  inline void parallel_for( std::string                    str    ,
                            Bounds<N,Style,simple> const & bounds ,
                            F                      const & f      ,
                            Config<MaxThreadsPerBlock,Strip> = Config<0,1>{} ) {
    using unsigned_t = size_t;
    using signed_t   = ptrdiff_t;
    using Kokkos::RangePolicy;
    using Kokkos::LaunchBounds;
    if (bounds.nIter == 0) return;  // exit early if there is no work to do
    if constexpr (yakl_auto_profile) timer_start(str);
    Kokkos::parallel_for( str ,
                          RangePolicy<LaunchBounds<MaxThreadsPerBlock,0>>(0,(bounds.nIter-1)/Strip+1) ,
                          KOKKOS_LAMBDA (size_t iglob) {
      auto &bloc = bounds;
      auto &floc = f;
      auto &nIter = bounds.nIter;
      if constexpr (Strip == 1) {
        if constexpr (simple) {
          if constexpr (N==1) {
            unsigned_t i0; bloc.unpack(iglob,i0); floc(i0);
          } else if constexpr (N==2) {
            unsigned_t i0,i1; bloc.unpack(iglob,i0,i1); floc(i0,i1);
          } else if constexpr (N==3) {
            unsigned_t i0,i1,i2; bloc.unpack(iglob,i0,i1,i2); floc(i0,i1,i2);
          } else if constexpr (N==4) {
            unsigned_t i0,i1,i2,i3; bloc.unpack(iglob,i0,i1,i2,i3); floc(i0,i1,i2,i3);
          } else if constexpr (N==5) {
            unsigned_t i0,i1,i2,i3,i4; bloc.unpack(iglob,i0,i1,i2,i3,i4); floc(i0,i1,i2,i3,i4);
          } else if constexpr (N==6) {
            unsigned_t i0,i1,i2,i3,i4,i5; bloc.unpack(iglob,i0,i1,i2,i3,i4,i5); floc(i0,i1,i2,i3,i4,i5);
          } else if constexpr (N==7) {
            unsigned_t i0,i1,i2,i3,i4,i5,i6; bloc.unpack(iglob,i0,i1,i2,i3,i4,i5,i6); floc(i0,i1,i2,i3,i4,i5,i6);
          } else if constexpr (N==8) {
            unsigned_t i0,i1,i2,i3,i4,i5,i6,i7; bloc.unpack(iglob,i0,i1,i2,i3,i4,i5,i6,i7); floc(i0,i1,i2,i3,i4,i5,i6,i7);
          }
        } else {
          if constexpr (N==1) {
            signed_t i0; bloc.unpack(iglob,i0); floc(i0);
          } else if constexpr (N==2) {
            signed_t i0,i1; bloc.unpack(iglob,i0,i1); floc(i0,i1);
          } else if constexpr (N==3) {
            signed_t i0,i1,i2; bloc.unpack(iglob,i0,i1,i2); floc(i0,i1,i2);
          } else if constexpr (N==4) {
            signed_t i0,i1,i2,i3; bloc.unpack(iglob,i0,i1,i2,i3); floc(i0,i1,i2,i3);
          } else if constexpr (N==5) {
            signed_t i0,i1,i2,i3,i4; bloc.unpack(iglob,i0,i1,i2,i3,i4); floc(i0,i1,i2,i3,i4);
          } else if constexpr (N==6) {
            signed_t i0,i1,i2,i3,i4,i5; bloc.unpack(iglob,i0,i1,i2,i3,i4,i5); floc(i0,i1,i2,i3,i4,i5);
          } else if constexpr (N==7) {
            signed_t i0,i1,i2,i3,i4,i5,i6; bloc.unpack(iglob,i0,i1,i2,i3,i4,i5,i6); floc(i0,i1,i2,i3,i4,i5,i6);
          } else if constexpr (N==8) {
            signed_t i0,i1,i2,i3,i4,i5,i6,i7; bloc.unpack(iglob,i0,i1,i2,i3,i4,i5,i6,i7); floc(i0,i1,i2,i3,i4,i5,i6,i7);
          }
        }
      } else {
        for (size_t i=0; i < Strip; i++) {
          size_t chunks = (bounds.nIter-1)/Strip+1;
          size_t iglob_real = iglob+i*chunks;
          if (iglob_real < nIter) {
            if constexpr (simple) {
              if constexpr (N==1) {
                unsigned_t i0; bloc.unpack(iglob_real,i0); floc(i0);
              } else if constexpr (N==2) {
                unsigned_t i0,i1; bloc.unpack(iglob_real,i0,i1); floc(i0,i1);
              } else if constexpr (N==3) {
                unsigned_t i0,i1,i2; bloc.unpack(iglob_real,i0,i1,i2); floc(i0,i1,i2);
              } else if constexpr (N==4) {
                unsigned_t i0,i1,i2,i3; bloc.unpack(iglob_real,i0,i1,i2,i3); floc(i0,i1,i2,i3);
              } else if constexpr (N==5) {
                unsigned_t i0,i1,i2,i3,i4; bloc.unpack(iglob_real,i0,i1,i2,i3,i4); floc(i0,i1,i2,i3,i4);
              } else if constexpr (N==6) {
                unsigned_t i0,i1,i2,i3,i4,i5; bloc.unpack(iglob_real,i0,i1,i2,i3,i4,i5); floc(i0,i1,i2,i3,i4,i5);
              } else if constexpr (N==7) {
                unsigned_t i0,i1,i2,i3,i4,i5,i6; bloc.unpack(iglob_real,i0,i1,i2,i3,i4,i5,i6); floc(i0,i1,i2,i3,i4,i5,i6);
              } else if constexpr (N==8) {
                unsigned_t i0,i1,i2,i3,i4,i5,i6,i7; bloc.unpack(iglob_real,i0,i1,i2,i3,i4,i5,i6,i7); floc(i0,i1,i2,i3,i4,i5,i6,i7);
              }
            } else {
              if constexpr (N==1) {
                signed_t i0; bloc.unpack(iglob_real,i0); floc(i0);
              } else if constexpr (N==2) {
                signed_t i0,i1; bloc.unpack(iglob_real,i0,i1); floc(i0,i1);
              } else if constexpr (N==3) {
                signed_t i0,i1,i2; bloc.unpack(iglob_real,i0,i1,i2); floc(i0,i1,i2);
              } else if constexpr (N==4) {
                signed_t i0,i1,i2,i3; bloc.unpack(iglob_real,i0,i1,i2,i3); floc(i0,i1,i2,i3);
              } else if constexpr (N==5) {
                signed_t i0,i1,i2,i3,i4; bloc.unpack(iglob_real,i0,i1,i2,i3,i4); floc(i0,i1,i2,i3,i4);
              } else if constexpr (N==6) {
                signed_t i0,i1,i2,i3,i4,i5; bloc.unpack(iglob_real,i0,i1,i2,i3,i4,i5); floc(i0,i1,i2,i3,i4,i5);
              } else if constexpr (N==7) {
                signed_t i0,i1,i2,i3,i4,i5,i6; bloc.unpack(iglob_real,i0,i1,i2,i3,i4,i5,i6); floc(i0,i1,i2,i3,i4,i5,i6);
              } else if constexpr (N==8) {
                signed_t i0,i1,i2,i3,i4,i5,i6,i7; bloc.unpack(iglob_real,i0,i1,i2,i3,i4,i5,i6,i7); floc(i0,i1,i2,i3,i4,i5,i6,i7);
              }
            }
          }
        }
      }
    });
    if constexpr (yakl_auto_profile) timer_stop(str);
    if constexpr (yakl_auto_fence) Kokkos::fence();
  }

  template <class F, int N, bool simple, class Style, int MaxThreadsPerBlock=0, size_t Strip=1>
  inline void parallel_for( Bounds<N,Style,simple> const & bounds , F const & f ,
                            Config<MaxThreadsPerBlock,Strip> config = Config<0,1>{} ) {
    parallel_for( YAKL_AUTO_LABEL() , bounds , f , config );
  }

  template <class F, int MaxThreadsPerBlock=0, size_t Strip=1>
  inline void parallel_for( std::integral auto bnd , F const & f ,
                            Config<MaxThreadsPerBlock,Strip> config = Config<0,1>{} ) {
    parallel_for( YAKL_AUTO_LABEL() , Bounds<1,CStyle,true>(bnd) , f , config );
  }

  template <class F, int MaxThreadsPerBlock=0, size_t Strip=1>
  inline void parallel_for( std::string str , std::integral auto bnd , F const & f ,
                            Config<MaxThreadsPerBlock,Strip> config = Config<0,1>{} ) {
    parallel_for( str , Bounds<1,CStyle,true>(bnd) , f , config );
  }



  template <class F, int N, bool simple, int MaxThreadsPerBlock=0, size_t Strip=1>
  inline void parallel_for_F( std::string str , Bounds<N,FStyle,simple> const & bounds , F const & f ,
                            Config<MaxThreadsPerBlock,Strip> config = Config<0,1>{} ) {
    parallel_for<F,N,simple,FStyle>( str , bounds , f , config );
  }

  template <class F, int N, bool simple, int MaxThreadsPerBlock=0, size_t Strip=1>
  inline void parallel_for_F( Bounds<N,FStyle,simple> const & bounds , F const & f ,
                            Config<MaxThreadsPerBlock,Strip> config = Config<0,1>{} ) {
    parallel_for<F,N,simple,FStyle>( YAKL_AUTO_LABEL() , bounds , f , config );
  }

  template <class F, int MaxThreadsPerBlock=0, size_t Strip=1>
  inline void parallel_for_F( std::integral auto bnd , F const & f ,
                            Config<MaxThreadsPerBlock,Strip> config = Config<0,1>{} ) {
    parallel_for<F,1,true,FStyle>( YAKL_AUTO_LABEL() , Bounds<1,FStyle,true>(bnd) , f , config );
  }

  template <class F, int MaxThreadsPerBlock=0, size_t Strip=1>
  inline void parallel_for_F( std::string str , std::integral auto bnd , F const & f ,
                            Config<MaxThreadsPerBlock,Strip> config = Config<0,1>{} ) {
    parallel_for<F,1,true,FStyle>( str , Bounds<1,FStyle,true>(bnd) , f , config );
  }

}

