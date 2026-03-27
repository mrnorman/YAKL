
#include "YAKL.h"

#ifdef KOKKOS_ENABLE_DEBUG
    inline constexpr bool kokkos_debug = true;
#else
    inline constexpr bool kokkos_debug = false;
#endif

#ifdef KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
    inline constexpr bool kokkos_bounds_debug = true;
#else
    inline constexpr bool kokkos_bounds_debug = false;
#endif


template <class Type> inline constexpr bool is_CSArray = requires { Type::is_CSArray; };
template <class Type> inline constexpr bool is_FSArray = requires { Type::is_FSArray; };

struct CStyle { static constexpr bool is_cstyle = true; using layout = Kokkos::LayoutRight; };
struct FStyle { static constexpr bool is_fstyle = true; using layout = Kokkos::LayoutLeft ; };

template <class Type> inline constexpr bool is_CStyle = requires { Type::is_cstyle; };
template <class Type> inline constexpr bool is_FStyle = requires { Type::is_fstyle; };

inline int constexpr COLON = 0;

struct FBounds { ptrdiff_t l, u; };



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



template <class Style = CStyle> class LBnd {
public:
  bool      static constexpr is_cstyle = is_CStyle<Style>;
  bool      static constexpr is_fstyle = is_FStyle<Style>;
  ptrdiff_t static constexpr default_lbound = is_cstyle ? 0 : 1;
  ptrdiff_t l, u, s;
  KOKKOS_INLINE_FUNCTION LBnd() : l(-1),u(-1),s(-1) { }
  KOKKOS_INLINE_FUNCTION LBnd(std::integral auto u) : l(default_lbound),u(u-1+default_lbound),s(1) { }
  KOKKOS_INLINE_FUNCTION LBnd(std::integral auto l, std::integral auto u) : l(l),u(u),s(1) {
    if constexpr (kokkos_debug) { if (u < l) Kokkos::abort("ERROR: cannot specify an upper bound < lower bound"); }
  }
  KOKKOS_INLINE_FUNCTION LBnd(std::integral auto l, std::integral auto u, std::integral auto s) : l(l),u(u),s(s) {
    if constexpr (kokkos_debug) { if (u < l) Kokkos::abort("ERROR: cannot specify an upper bound < lower bound"); }
    if constexpr (kokkos_debug) { if (s < 1) Kokkos::abort("ERROR: non-positive strides not supported."); }
  }
  KOKKOS_INLINE_FUNCTION bool valid() const { return this->s > 0; }
};



template <int N, class Style = CStyle, bool Simple = false> class Bounds;

template<int N, class Style> class Bounds<N,Style,true> {
  public:
  bool   static constexpr is_cstyle = is_CStyle<Style>;
  bool   static constexpr is_fstyle = is_FStyle<Style>;
  size_t static constexpr default_lbound = is_cstyle ? 0 : 1;
  size_t nIter;
  std::array<size_t,N> offs;
  KOKKOS_INLINE_FUNCTION Bounds( std::integral auto... sizes ) requires (sizeof...(sizes) == N) {
    std::array<size_t,N> dims = { static_cast<size_t>(sizes)... };
    nIter = 1;
    for (int i=0; i < N; i++) {
      nIter *= dims[i];
      offs[i] = 1;
      for (int j=i+1; j < N; j++) { offs[i] *= dims[j]; }
    }
  }
  KOKKOS_INLINE_FUNCTION void unpackIndices( size_t iGlob , size_t & i0 ) const requires (N==1) {
    i0 = iGlob        ;                        i0 += default_lbound;
  }
  KOKKOS_INLINE_FUNCTION void unpackIndices( size_t iGlob , size_t & i0 , size_t & i1 ) const requires (N==2) {
    i0 = iGlob/offs[0];  iGlob -= offs[0]*i0;  i0 += default_lbound;
    i1 = iGlob        ;                        i1 += default_lbound;
  }
  KOKKOS_INLINE_FUNCTION void unpackIndices( size_t iGlob , size_t & i0 , size_t & i1 , size_t & i2 ) const requires (N==3) {
    i0 = iGlob/offs[0];  iGlob -= offs[0]*i0;  i0 += default_lbound;
    i1 = iGlob/offs[1];  iGlob -= offs[1]*i1;  i1 += default_lbound;
    i2 = iGlob        ;                        i2 += default_lbound;
  }
  KOKKOS_INLINE_FUNCTION void unpackIndices( size_t iGlob , size_t & i0 , size_t & i1 , size_t & i2 , size_t & i3 ) const requires (N==4) {
    i0 = iGlob/offs[0];  iGlob -= offs[0]*i0;  i0 += default_lbound;
    i1 = iGlob/offs[1];  iGlob -= offs[1]*i1;  i1 += default_lbound;
    i2 = iGlob/offs[2];  iGlob -= offs[2]*i2;  i2 += default_lbound;
    i3 = iGlob        ;                        i3 += default_lbound;
  }
  KOKKOS_INLINE_FUNCTION void unpackIndices( size_t iGlob , size_t & i0 , size_t & i1 , size_t & i2 , size_t & i3 , size_t & i4) const requires (N==5) {
    i0 = iGlob/offs[0];  iGlob -= offs[0]*i0;  i0 += default_lbound;
    i1 = iGlob/offs[1];  iGlob -= offs[1]*i1;  i1 += default_lbound;
    i2 = iGlob/offs[2];  iGlob -= offs[2]*i2;  i2 += default_lbound;
    i3 = iGlob/offs[3];  iGlob -= offs[3]*i3;  i3 += default_lbound;
    i4 = iGlob        ;                        i4 += default_lbound;
  }
  KOKKOS_INLINE_FUNCTION void unpackIndices( size_t iGlob , size_t & i0 , size_t & i1 , size_t & i2 , size_t & i3 , size_t & i4 , size_t & i5) const requires (N==6) {
    i0 = iGlob/offs[0];  iGlob -= offs[0]*i0;  i0 += default_lbound;
    i1 = iGlob/offs[1];  iGlob -= offs[1]*i1;  i1 += default_lbound;
    i2 = iGlob/offs[2];  iGlob -= offs[2]*i2;  i2 += default_lbound;
    i3 = iGlob/offs[3];  iGlob -= offs[3]*i3;  i3 += default_lbound;
    i4 = iGlob/offs[4];  iGlob -= offs[4]*i4;  i4 += default_lbound;
    i5 = iGlob        ;                        i5 += default_lbound;
  }
  KOKKOS_INLINE_FUNCTION void unpackIndices( size_t iGlob , size_t & i0 , size_t & i1 , size_t & i2 , size_t & i3 , size_t & i4 , size_t & i5 , size_t & i6) const requires (N==7) {
    i0 = iGlob/offs[0];  iGlob -= offs[0]*i0;  i0 += default_lbound;
    i1 = iGlob/offs[1];  iGlob -= offs[1]*i1;  i1 += default_lbound;
    i2 = iGlob/offs[2];  iGlob -= offs[2]*i2;  i2 += default_lbound;
    i3 = iGlob/offs[3];  iGlob -= offs[3]*i3;  i3 += default_lbound;
    i4 = iGlob/offs[4];  iGlob -= offs[4]*i4;  i4 += default_lbound;
    i5 = iGlob/offs[5];  iGlob -= offs[5]*i5;  i5 += default_lbound;
    i6 = iGlob        ;                        i6 += default_lbound;
  }
  KOKKOS_INLINE_FUNCTION void unpackIndices( size_t iGlob , size_t & i0 , size_t & i1 , size_t & i2 , size_t & i3 , size_t & i4 , size_t & i5 , size_t & i6 , size_t & i7) const requires (N==8) {
    i0 = iGlob/offs[0];  iGlob -= offs[0]*i0;  i0 += default_lbound;
    i1 = iGlob/offs[1];  iGlob -= offs[1]*i1;  i1 += default_lbound;
    i2 = iGlob/offs[2];  iGlob -= offs[2]*i2;  i2 += default_lbound;
    i3 = iGlob/offs[3];  iGlob -= offs[3]*i3;  i3 += default_lbound;
    i4 = iGlob/offs[4];  iGlob -= offs[4]*i4;  i4 += default_lbound;
    i5 = iGlob/offs[5];  iGlob -= offs[5]*i5;  i5 += default_lbound;
    i6 = iGlob/offs[6];  iGlob -= offs[6]*i6;  i6 += default_lbound;
    i7 = iGlob        ;                        i7 += default_lbound;
  }
};

template<int N, class Style> class Bounds<N,Style,false> {
  public:
  bool   static constexpr is_cstyle = is_CStyle<Style>;
  bool   static constexpr is_fstyle = is_FStyle<Style>;
  size_t nIter;

  std::array<size_t   ,N> offs;
  std::array<ptrdiff_t,N> lbounds;
  std::array<size_t   ,N> strides;

  template <class... BNDS> requires (std::is_same_v<BNDS,LBnd<Style>> && ...)
  KOKKOS_INLINE_FUNCTION void init( BNDS... bnds ) requires (sizeof...(bnds) == N) {
    std::array<size_t,N> dims = { static_cast<size_t   >((bnds.u-bnds.l+1)/bnds.s)... };
    lbounds                   = { static_cast<ptrdiff_t>(bnds.l)... };
    strides                   = { static_cast<size_t   >(bnds.s)... };
    nIter = 1;
    for (int i=0; i < N; i++) {
      nIter *= dims[i];
      offs[i] = 1;
      for (int j=i+1; j < N; j++) { offs[i] *= dims[j]; }
    }
  }

  using LB = LBnd<Style>;
  KOKKOS_INLINE_FUNCTION Bounds(LB b0)                                           requires (N==1) { init(b0); }
  KOKKOS_INLINE_FUNCTION Bounds(LB b0,LB b1)                                     requires (N==2) { init(b0,b1); }
  KOKKOS_INLINE_FUNCTION Bounds(LB b0,LB b1,LB b2)                               requires (N==3) { init(b0,b1,b2); }
  KOKKOS_INLINE_FUNCTION Bounds(LB b0,LB b1,LB b2,LB b3)                         requires (N==4) { init(b0,b1,b2,b3); }
  KOKKOS_INLINE_FUNCTION Bounds(LB b0,LB b1,LB b2,LB b3,LB b4)                   requires (N==5) { init(b0,b1,b2,b3,b4); }
  KOKKOS_INLINE_FUNCTION Bounds(LB b0,LB b1,LB b2,LB b3,LB b4,LB b5)             requires (N==6) { init(b0,b1,b2,b3,b4,b5); }
  KOKKOS_INLINE_FUNCTION Bounds(LB b0,LB b1,LB b2,LB b3,LB b4,LB b5,LB b6)       requires (N==7) { init(b0,b1,b2,b3,b4,b5,b6); }
  KOKKOS_INLINE_FUNCTION Bounds(LB b0,LB b1,LB b2,LB b3,LB b4,LB b5,LB b6,LB b7) requires (N==8) { init(b0,b1,b2,b3,b4,b5,b6,b7); }

  KOKKOS_INLINE_FUNCTION void unpackIndices( size_t iGlob , ptrdiff_t & i0 ) const requires (N==1) {
    i0 = iGlob        ;                        i0 = i0*strides[0]+lbounds[0];
  }
  KOKKOS_INLINE_FUNCTION void unpackIndices( size_t iGlob , ptrdiff_t & i0 , ptrdiff_t & i1 ) const requires (N==2) {
    i0 = iGlob/offs[0];  iGlob -= offs[0]*i0;  i0 = i0*strides[0]+lbounds[0];
    i1 = iGlob        ;                        i1 = i1*strides[1]+lbounds[1];
  }
  KOKKOS_INLINE_FUNCTION void unpackIndices( size_t iGlob , ptrdiff_t & i0 , ptrdiff_t & i1 , ptrdiff_t & i2 ) const requires (N==3) {
    i0 = iGlob/offs[0];  iGlob -= offs[0]*i0;  i0 = i0*strides[0]+lbounds[0];
    i1 = iGlob/offs[1];  iGlob -= offs[1]*i1;  i1 = i1*strides[1]+lbounds[1];
    i2 = iGlob        ;                        i2 = i2*strides[2]+lbounds[2];
  }
  KOKKOS_INLINE_FUNCTION void unpackIndices( size_t iGlob , ptrdiff_t & i0 , ptrdiff_t & i1 , ptrdiff_t & i2 , ptrdiff_t & i3 ) const requires (N==4) {
    i0 = iGlob/offs[0];  iGlob -= offs[0]*i0;  i0 = i0*strides[0]+lbounds[0];
    i1 = iGlob/offs[1];  iGlob -= offs[1]*i1;  i1 = i1*strides[1]+lbounds[1];
    i2 = iGlob/offs[2];  iGlob -= offs[2]*i2;  i2 = i2*strides[2]+lbounds[2];
    i3 = iGlob        ;                        i3 = i3*strides[3]+lbounds[3];
  }
  KOKKOS_INLINE_FUNCTION void unpackIndices( size_t iGlob , ptrdiff_t & i0 , ptrdiff_t & i1 , ptrdiff_t & i2 , ptrdiff_t & i3 , ptrdiff_t & i4) const requires (N==5) {
    i0 = iGlob/offs[0];  iGlob -= offs[0]*i0;  i0 = i0*strides[0]+lbounds[0];
    i1 = iGlob/offs[1];  iGlob -= offs[1]*i1;  i1 = i1*strides[1]+lbounds[1];
    i2 = iGlob/offs[2];  iGlob -= offs[2]*i2;  i2 = i2*strides[2]+lbounds[2];
    i3 = iGlob/offs[3];  iGlob -= offs[3]*i3;  i3 = i3*strides[3]+lbounds[3];
    i4 = iGlob        ;                        i4 = i4*strides[4]+lbounds[4];
  }
  KOKKOS_INLINE_FUNCTION void unpackIndices( size_t iGlob , ptrdiff_t & i0 , ptrdiff_t & i1 , ptrdiff_t & i2 , ptrdiff_t & i3 , ptrdiff_t & i4 , ptrdiff_t & i5) const requires (N==6) {
    i0 = iGlob/offs[0];  iGlob -= offs[0]*i0;  i0 = i0*strides[0]+lbounds[0];
    i1 = iGlob/offs[1];  iGlob -= offs[1]*i1;  i1 = i1*strides[1]+lbounds[1];
    i2 = iGlob/offs[2];  iGlob -= offs[2]*i2;  i2 = i2*strides[2]+lbounds[2];
    i3 = iGlob/offs[3];  iGlob -= offs[3]*i3;  i3 = i3*strides[3]+lbounds[3];
    i4 = iGlob/offs[4];  iGlob -= offs[4]*i4;  i4 = i4*strides[4]+lbounds[4];
    i5 = iGlob        ;                        i5 = i5*strides[5]+lbounds[5];
  }
  KOKKOS_INLINE_FUNCTION void unpackIndices( size_t iGlob , ptrdiff_t & i0 , ptrdiff_t & i1 , ptrdiff_t & i2 , ptrdiff_t & i3 , ptrdiff_t & i4 , ptrdiff_t & i5 , ptrdiff_t & i6) const requires (N==7) {
    i0 = iGlob/offs[0];  iGlob -= offs[0]*i0;  i0 = i0*strides[0]+lbounds[0];
    i1 = iGlob/offs[1];  iGlob -= offs[1]*i1;  i1 = i1*strides[1]+lbounds[1];
    i2 = iGlob/offs[2];  iGlob -= offs[2]*i2;  i2 = i2*strides[2]+lbounds[2];
    i3 = iGlob/offs[3];  iGlob -= offs[3]*i3;  i3 = i3*strides[3]+lbounds[3];
    i4 = iGlob/offs[4];  iGlob -= offs[4]*i4;  i4 = i4*strides[4]+lbounds[4];
    i5 = iGlob/offs[5];  iGlob -= offs[5]*i5;  i5 = i5*strides[5]+lbounds[5];
    i6 = iGlob        ;                        i6 = i6*strides[6]+lbounds[6];
  }
  KOKKOS_INLINE_FUNCTION void unpackIndices( size_t iGlob , ptrdiff_t & i0 , ptrdiff_t & i1 , ptrdiff_t & i2 , ptrdiff_t & i3 , ptrdiff_t & i4 , ptrdiff_t & i5 , ptrdiff_t & i6 , ptrdiff_t & i7) const requires (N==8) {
    i0 = iGlob/offs[0];  iGlob -= offs[0]*i0;  i0 = i0*strides[0]+lbounds[0];
    i1 = iGlob/offs[1];  iGlob -= offs[1]*i1;  i1 = i1*strides[1]+lbounds[1];
    i2 = iGlob/offs[2];  iGlob -= offs[2]*i2;  i2 = i2*strides[2]+lbounds[2];
    i3 = iGlob/offs[3];  iGlob -= offs[3]*i3;  i3 = i3*strides[3]+lbounds[3];
    i4 = iGlob/offs[4];  iGlob -= offs[4]*i4;  i4 = i4*strides[4]+lbounds[4];
    i5 = iGlob/offs[5];  iGlob -= offs[5]*i5;  i5 = i5*strides[5]+lbounds[5];
    i6 = iGlob/offs[6];  iGlob -= offs[6]*i6;  i6 = i6*strides[6]+lbounds[6];
    i7 = iGlob        ;                        i7 = i7*strides[7]+lbounds[7];
  }
};



template <class T, std::size_t... DIMS> requires (sizeof...(DIMS) > 0) && ((DIMS > 0) && ...)
class CSArray {
  public:
  bool                    static constexpr is_CSArray    = true;
  size_t                  static constexpr rank          = sizeof...(DIMS);
  size_t                  static constexpr dims[rank]    = {DIMS...};
  size_t                  static constexpr num_elements  = (DIMS * ...);
  std::array<size_t,rank> static constexpr offsets       = [] {
    std::array<size_t,rank> result = {};
    for (int i=0; i < static_cast<int>(rank); i++) {
      result[i] = 1;
      for (int j = i+1; j < static_cast<int>(rank); j++) result[i] *= dims[j];
    }
    return result;
  }();
  using value_type           = T;
  using const_value_type     = std::add_const_t<T>;
  using non_const_value_type = std::remove_cv_t<T>;

  T mutable my_data[num_elements];

  template <class TLOC> requires std::is_arithmetic_v<TLOC>
  KOKKOS_INLINE_FUNCTION void operator= (TLOC val) { for (size_t i=0; i < size(); i++) { my_data[i] = val; } }

  KOKKOS_INLINE_FUNCTION T & operator()(std::integral auto... indices) const {
    static_assert( sizeof...(indices) == rank , "ERROR: Indexing CSArray with the wrong number of indices" );
    size_t idx[rank] = {static_cast<size_t>(indices)...};
    size_t offset = 0;
    for (int i = 0; i < rank; i++) offset += idx[i] * offsets[i];
    if constexpr (kokkos_bounds_debug) {
      for (int i = 0; i < rank; i++) {
        if (idx[i] >= dims[i]) Kokkos::abort("ERROR: CSArray index out of bounds");
      }
    }
    return my_data[offset];
  }

  KOKKOS_INLINE_FUNCTION T * data () const { return my_data; }
  KOKKOS_INLINE_FUNCTION T * begin() const { return my_data; }
  KOKKOS_INLINE_FUNCTION T * end  () const { return my_data + size(); }
  KOKKOS_INLINE_FUNCTION size_t static constexpr size() { return num_elements; }
  KOKKOS_INLINE_FUNCTION bool   static constexpr span_is_contiguous() { return true; }
  KOKKOS_INLINE_FUNCTION size_t static constexpr extent(std::integral auto i) {
    if constexpr (kokkos_debug) {
      if ((std::is_signed_v<decltype(i)> && i < 0) || static_cast<size_t>(i) >= rank) {
        Kokkos::abort("ERROR: calling CSArray extent() with out of bounds index"); 
      }
    }
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
    for (size_t i=0; i < rank; i++) { ret(i) = dims[i]; }
    return ret;
  }

  KOKKOS_INLINE_FUNCTION auto lbounds() const {
    CSArray<size_t,rank> ret;
    for (size_t i=0; i < rank; i++) { ret(i) = 0; }
    return ret;
  }

  KOKKOS_INLINE_FUNCTION auto ubounds() const {
    CSArray<size_t,rank> ret;
    for (size_t i=0; i < rank; i++) { ret(i) = dims[i]-1; }
    return ret;
  }
};



template <class T, FBounds... DIMS> requires (sizeof...(DIMS) > 0) &&
                                             ((static_cast<ptrdiff_t>(DIMS.l) <= static_cast<ptrdiff_t>(DIMS.u)) && ...)
class FSArray {
  public:
  bool                    static constexpr is_FSArray   = true;
  size_t                  static constexpr rank         = sizeof...(DIMS);
  ptrdiff_t               static constexpr lb  [rank]   = {static_cast<ptrdiff_t>(DIMS.l)...};
  ptrdiff_t               static constexpr ub  [rank]   = {static_cast<ptrdiff_t>(DIMS.u)...};
  size_t                  static constexpr dims[rank]   = {(static_cast<size_t>(static_cast<ptrdiff_t>(DIMS.u)-static_cast<ptrdiff_t>(DIMS.l)+1))...};
  size_t                  static constexpr num_elements = ((static_cast<size_t>(static_cast<ptrdiff_t>(DIMS.u)-static_cast<ptrdiff_t>(DIMS.l)+1)) * ...);
  std::array<size_t,rank> static constexpr offsets      = [] {
    std::array<size_t,rank> result = {};
    for (int i=static_cast<int>(rank)-1; i >= 0; i--) {
      result[i] = 1;
      for (int j = i-1; j >= 0; j--) result[i] *= dims[j];
    }
    return result;
  }();
  using value_type           = T;
  using const_value_type     = std::add_const_t<T>;
  using non_const_value_type = std::remove_cv_t<T>;

  T mutable my_data[num_elements];

  template <class TLOC> requires std::is_arithmetic_v<TLOC>
  KOKKOS_INLINE_FUNCTION void operator= (TLOC val) { for (size_t i=0; i < size(); i++) { my_data[i] = val; } }

  KOKKOS_INLINE_FUNCTION T & operator()(std::integral auto... indices) const {
    static_assert( sizeof...(indices) == rank , "ERROR: Indexing FSArray with the wrong number of indices" );
    ptrdiff_t idx[rank] = {static_cast<ptrdiff_t>(indices)...};
    size_t offset = 0;
    for (int i = 0; i < rank; i++) offset += (idx[i]-lb[i]) * offsets[i];
    if constexpr (kokkos_bounds_debug) {
      for (int i = 0; i < rank; i++) {
        if (idx[i] > ub[i] || idx[i] < lb[i]) Kokkos::abort("ERROR: FSArray index out of bounds");
      }
    }
    return my_data[offset];
  }

  KOKKOS_INLINE_FUNCTION T * data () const { return my_data; }
  KOKKOS_INLINE_FUNCTION T * begin() const { return my_data; }
  KOKKOS_INLINE_FUNCTION T * end  () const { return my_data + size(); }
  KOKKOS_INLINE_FUNCTION size_t static constexpr size() { return num_elements; }
  KOKKOS_INLINE_FUNCTION bool   static constexpr span_is_contiguous() { return true; }
  KOKKOS_INLINE_FUNCTION size_t static constexpr extent(std::integral auto i) {
    if constexpr (kokkos_debug) {
      if ((std::is_signed_v<decltype(i)> && i < 0) || static_cast<size_t>(i) >= rank) {
        Kokkos::abort("ERROR: calling FSArray extent() with out of bounds index"); 
      }
    }
    return dims[i];
  }

  inline friend std::ostream &operator<<( std::ostream& os , FSArray const & v ) {
    os << "yakl::FSArray: ";
    for (size_t i = 0; i < size(); i++) { os << v.my_data[i] << (i<size()-1 ? " , " : ""); }
    os << std::endl;
    return os;
  }

  KOKKOS_INLINE_FUNCTION auto extents() const {
    FSArray<size_t,{1,rank}> ret;
    for (size_t i=1; i <= rank; i++) { ret(i) = dims[i-1]; }
    return ret;
  }

  KOKKOS_INLINE_FUNCTION auto lbounds() const {
    FSArray<ptrdiff_t,{1,rank}> ret;
    for (size_t i=1; i <= rank; i++) { ret(i) = lb[i-1]; }
    return ret;
  }

  KOKKOS_INLINE_FUNCTION auto ubounds() const {
    FSArray<ptrdiff_t,{1,rank}> ret;
    for (size_t i=1; i <= rank; i++) { ret(i) = ub[i-1]; }
    return ret;
  }
};



template <class KT, class MemSpace, class Style = CStyle>
class Array : public Kokkos::View<KT,typename Style::layout,MemSpace> {

  template <typename T, int N> struct KokkosType { using type = typename KokkosType<T*,N-1>::type; };
  template <typename T> struct KokkosType<T,0> { using type = T; };

  public:

  using base_t = Kokkos::View<KT,typename Style::layout,MemSpace>;
  using this_t = Array<KT,MemSpace,Style>;
  using base_t::base_t; 
  using base_t::operator=;
  using base_t::operator();

  bool static constexpr is_fstyle = is_FStyle<Style>;
  bool static constexpr is_cstyle = is_CStyle<Style>;

  // If this is FStyle, create a std::array that initializes statically to zero of size rank
  // If this is CStyle, create a zero-size array with no memory in the class and no compiler warnings
  [[no_unique_address]] std::array< ptrdiff_t , is_fstyle ? base_t::rank : 0 > lb = {};


  // Fortran-style constructors
  // This gets ugly when trying to use parameter packs, so I went with brute force here
  Array(std::string const & label, FBounds b1)
      requires is_fstyle && (base_t::rank==1)
      : base_t(label,b1.u-b1.l+1) ,
        lb({b1.l}) {}
  Array(std::string const & label, FBounds b1, FBounds b2)
      requires is_fstyle && (base_t::rank==2)
      : base_t(label,b1.u-b1.l+1,b2.u-b2.l+1) ,
        lb({b1.l,b2.l}) {}
  Array(std::string const & label, FBounds b1, FBounds b2, FBounds b3)
      requires is_fstyle && (base_t::rank==3)
      : base_t(label,b1.u-b1.l+1,b2.u-b2.l+1,b3.u-b3.l+1) ,
        lb({b1.l,b2.l,b3.l}) {}
  Array(std::string const & label, FBounds b1, FBounds b2, FBounds b3, FBounds b4)
      requires is_fstyle && (base_t::rank==4)
      : base_t(label,b1.u-b1.l+1,b2.u-b2.l+1,b3.u-b3.l+1,b4.u-b4.l+1) ,
        lb({b1.l,b2.l,b3.l,b4.l}) {}
  Array(std::string const & label, FBounds b1, FBounds b2, FBounds b3, FBounds b4, FBounds b5)
      requires is_fstyle && (base_t::rank==5)
      : base_t(label,b1.u-b1.l+1,b2.u-b2.l+1,b3.u-b3.l+1,b4.u-b4.l+1,b5.u-b5.l+1) ,
        lb({b1.l,b2.l,b3.l,b4.l,b5.l}) {}
  Array(std::string const & label, FBounds b1, FBounds b2, FBounds b3, FBounds b4, FBounds b5, FBounds b6)
      requires is_fstyle && (base_t::rank==6)
      : base_t(label,b1.u-b1.l+1,b2.u-b2.l+1,b3.u-b3.l+1,b4.u-b4.l+1,b5.u-b5.l+1,b6.u-b6.l+1) ,
        lb({b1.l,b2.l,b3.l,b4.l,b5.l,b6.l}) {}
  Array(std::string const & label, FBounds b1, FBounds b2, FBounds b3, FBounds b4, FBounds b5, FBounds b6, FBounds b7)
      requires is_fstyle && (base_t::rank==7)
      : base_t(label,b1.u-b1.l+1,b2.u-b2.l+1,b3.u-b3.l+1,b4.u-b4.l+1,b5.u-b5.l+1,b6.u-b6.l+1,b7.u-b7.l+1) ,
        lb({b1.l,b2.l,b3.l,b4.l,b5.l,b6.l,b7.l}) {}
  Array(std::string const & label, FBounds b1, FBounds b2, FBounds b3, FBounds b4, FBounds b5, FBounds b6, FBounds b7, FBounds b8)
      requires is_fstyle && (base_t::rank==8)
      : base_t(label,b1.u-b1.l+1,b2.u-b2.l+1,b3.u-b3.l+1,b4.u-b4.l+1,b5.u-b5.l+1,b6.u-b6.l+1,b7.u-b7.l+1,b8.u-b8.l+1) ,
        lb({b1.l,b2.l,b3.l,b4.l,b5.l,b6.l,b7.l,b8.l}) {}


  // Fortran-style operator()
  // I tried using parameter packs for this, but the compiler has trouble correctly overriding View operator()
  //   functions when using them, so I'm specifying each of these individually
  KOKKOS_INLINE_FUNCTION auto & operator()(std::integral auto i0) const
        requires is_fstyle && (base_t::rank == 1) {
    return base_t::operator()(i0-lb[0]);
  }
  KOKKOS_INLINE_FUNCTION auto & operator()(std::integral auto i0, std::integral auto i1) const
        requires is_fstyle && (base_t::rank == 2) {
    return base_t::operator()(i0-lb[0],i1-lb[1]);
  }
  KOKKOS_INLINE_FUNCTION auto & operator()(std::integral auto i0, std::integral auto i1, std::integral auto i2) const
        requires is_fstyle && (base_t::rank == 3) {
    return base_t::operator()(i0-lb[0],i1-lb[1],i2-lb[2]);
  }
  KOKKOS_INLINE_FUNCTION auto & operator()(std::integral auto i0, std::integral auto i1, std::integral auto i2,
                                           std::integral auto i3) const
        requires is_fstyle && (base_t::rank == 4) {
    return base_t::operator()(i0-lb[0],i1-lb[1],i2-lb[2],i3-lb[3]);
  }
  KOKKOS_INLINE_FUNCTION auto & operator()(std::integral auto i0, std::integral auto i1, std::integral auto i2,
                                           std::integral auto i3, std::integral auto i4) const
        requires is_fstyle && (base_t::rank == 5) {
    return base_t::operator()(i0-lb[0],i1-lb[1],i2-lb[2],i3-lb[3],i4-lb[4]);
  }
  KOKKOS_INLINE_FUNCTION auto & operator()(std::integral auto i0, std::integral auto i1, std::integral auto i2,
                                           std::integral auto i3, std::integral auto i4, std::integral auto i5) const
        requires is_fstyle && (base_t::rank == 6) {
    return base_t::operator()(i0-lb[0],i1-lb[1],i2-lb[2],i3-lb[3],i4-lb[4],i5-lb[5]);
  }
  KOKKOS_INLINE_FUNCTION auto & operator()(std::integral auto i0, std::integral auto i1, std::integral auto i2,
                                           std::integral auto i3, std::integral auto i4, std::integral auto i5,
                                           std::integral auto i6) const
        requires is_fstyle && (base_t::rank == 7) {
    return base_t::operator()(i0-lb[0],i1-lb[1],i2-lb[2],i3-lb[3],i4-lb[4],i5-lb[5],i6-lb[6]);
  }
  KOKKOS_INLINE_FUNCTION auto & operator()(std::integral auto i0, std::integral auto i1, std::integral auto i2,
                                           std::integral auto i3, std::integral auto i4, std::integral auto i5,
                                           std::integral auto i6, std::integral auto i7) const
        requires is_fstyle && (base_t::rank == 8) {
    return base_t::operator()(i0-lb[0],i1-lb[1],i2-lb[2],i3-lb[3],i4-lb[4],i5-lb[5],i6-lb[6],i7-lb[7]);
  }


  template <class TLOC> requires std::is_arithmetic_v<TLOC>
  Array const & operator=(TLOC const & v) const { Kokkos::deep_copy(*this,v); return *this; }


  template <class MemSpaceLoc>
  auto clone_object() const {
    auto func = [&] <std::size_t... Is> (std::index_sequence<Is...>) {
      auto loc = Array<typename base_t::non_const_data_type,MemSpaceLoc,Style>( this->label() , this->extent(Is)... );
      if constexpr (is_fstyle) loc.lb = this->lb;
      return loc;
    };
    return func(std::make_index_sequence<base_t::rank>{});
  }


  template <std::integral auto new_rank> requires (new_rank <= base_t::rank) && (new_rank >= 0)
  KOKKOS_INLINE_FUNCTION auto slice(std::integral auto... indices) requires (sizeof...(indices) == base_t::rank) {
    int constexpr nslice    = base_t::rank - new_rank;
    int constexpr remaining = new_rank;
    using new_kt = typename KokkosType<typename base_t::value_type,remaining>::type;
    size_t offset = 0;
    if constexpr (is_cstyle) {
      std::array<size_t,base_t::rank> slice_arr = { static_cast<size_t>(indices)... };
      for (int i=0; i < nslice; i++) { offset += slice_arr[i] * this->stride(i); }
      return [&] <std::size_t... Ir> ( std::index_sequence<Ir...> ) {
        return Array<new_kt,MemSpace,Style>( this->data()+offset , this->extent(nslice + Ir)... );
      } ( std::make_index_sequence<remaining>{} );
    } else {
      std::array<ptrdiff_t,base_t::rank> slice_arr = { static_cast<ptrdiff_t>(indices)... };
      for (int i=0; i < nslice; i++) {
        offset += (slice_arr[base_t::rank-1-i]-this->lb[base_t::rank-1-i]) * this->stride(base_t::rank-1-i);
      }
      return [&] <std::size_t... Ir> ( std::index_sequence<Ir...> ) {
        auto loc = Array<new_kt,MemSpace,Style>( this->data()+offset , this->extent(Ir)... );
        for (int i=0; i < remaining; i++) { loc.lb[i] = this->lb[i]; }
        return loc;
      } ( std::make_index_sequence<remaining>{} );
    }
  }


  KOKKOS_INLINE_FUNCTION auto reshape(std::integral auto... newdims) const {
    int constexpr new_rank = sizeof...(newdims);
    using new_kt = typename KokkosType<typename base_t::value_type,new_rank>::type;
    if ((static_cast<size_t>(newdims) * ...) != this->size()) {
      Kokkos::abort("ERROR: Resizing array with different total size");
    }
    auto loc = Array<new_kt,MemSpace,Style>(this->data(),newdims...);
    if constexpr (is_fstyle) loc.lb.fill(1);
    return loc;
  }


  KOKKOS_INLINE_FUNCTION auto collapse() const {
    auto loc = Array<typename KokkosType<typename base_t::value_type,1>::type,MemSpace,Style>(this->data(),this->size());
    if constexpr (is_fstyle) loc.lb.fill(1);
    return loc;
  }


  template <class ViewType>
  void deep_copy_to(ViewType const & them) const {
    static_assert(ViewType::is_fstyle == this_t::is_fstyle , "ERROR: calling deep_copy_to between Fortran and C style Arrays");
    if (them.size() != this->size()) Kokkos::abort("ERROR: calling deep_copy_to between differently sized arrays");
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
    auto func = [&] <std::size_t... Is> (std::index_sequence<Is...>) {
      return Array<typename KokkosType<scalar_t,base_t::rank>::type,MemSpace,Style>( this->label() , this->extent(Is)... );
    };
    auto ret = func(std::make_index_sequence<base_t::rank>{});
    YAKL_SCOPE( me , *this );
    Kokkos::parallel_for( "yakl_as_copy" ,
                          Kokkos::RangePolicy<typename base_t::execution_space>(0,this->size()) ,
                          KOKKOS_LAMBDA (int i) {
      ret.data()[i] = me.data()[i];
    });
    if constexpr (is_fstyle) for (int i=0; i < base_t::rank; i++) { ret.lb[i] = this->lb[i]; }
    return ret;
  }


  KOKKOS_INLINE_FUNCTION auto extents() const {
    if constexpr (is_cstyle) {
      CSArray<size_t,base_t::rank> ret;
      for (int i=0; i < base_t::rank; i++) { ret(i) = this->extent(i); }
      return ret;
    } else {
      FSArray<size_t,{1,static_cast<int>(base_t::rank)}> ret;
      for (int i=1; i <= base_t::rank; i++) { ret(i) = this->extent(i-1); }
      return ret;
    }
  }


  KOKKOS_INLINE_FUNCTION auto ubounds() const {
    if constexpr (is_cstyle) {
      CSArray<size_t,base_t::rank> ret;
      for (int i=0; i < base_t::rank; i++) { ret(i) = this->extent(i)-1; }
      return ret;
    } else {
      FSArray<ptrdiff_t,{1,static_cast<int>(base_t::rank)}> ret;
      for (int i=1; i <= base_t::rank; i++) { ret(i) = lb[i-1] + this->extent(i-1)-1; }
      return ret;
    }
  }


  KOKKOS_INLINE_FUNCTION auto lbounds() const {
    if constexpr (is_cstyle) {
      CSArray<size_t,base_t::rank> ret;
      for (int i=0; i < base_t::rank; i++) { ret(i) = 0; }
      return ret;
    } else {
      FSArray<ptrdiff_t,{1,static_cast<int>(base_t::rank)}> ret;
      for (int i=1; i <= base_t::rank; i++) { ret(i) = lb[i-1]; }
      return ret;
    }
  }


  KOKKOS_INLINE_FUNCTION base_t::value_type * begin() const { return this->data(); }
  KOKKOS_INLINE_FUNCTION base_t::value_type * end  () const { return this->data()+this->size(); }


  inline friend std::ostream &operator<<( std::ostream& os , Array const & v ) {
    auto loc = v.createHostCopy(); // cout,cerr is expensive, so just create a host copy
    os << "Array [" << loc.label() << "], Dimensions [";
    for (int i = 0; i < loc.rank(); i++) { os << loc.extent(i) << (i<loc.rank()-1 ? "," : ""); }
    os << "] = " << loc.size() << " Elements:  ";
    for (int i = 0; i < loc.size(); i++) { os << loc.data()[i] << (i<loc.size()-1 ? " , " : ""); }
    os << std::endl;
    return os;
  }


  base_t const & get_View() const { return static_cast<base_t const &>(*this); }
  base_t       & get_View()       { return static_cast<base_t       &>(*this); }
};



template <class ViewType>
inline typename ViewType::non_const_value_type sum(ViewType const & a) {
  if constexpr (kokkos_debug) {
    if (! a.span_is_contiguous()) Kokkos::abort("ERROR: Computing sum on non-contiguous View");
  }
  using scalar_t = typename ViewType::non_const_value_type;
  scalar_t result = 0;
  if constexpr (is_CSArray<ViewType> || is_FSArray<ViewType>) {
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
    Array<float *,yakl::PoolSpace> arr("arr",10);
    Kokkos::parallel_for( "mykernel" , 10 , KOKKOS_LAMBDA (int i) {
      arr(i) = i+1;
    });
    auto arr_h = arr.createHostObject();
    arr.deep_copy_to(arr_h);
    auto arr_d = arr_h.createDeviceCopy();
    std::cout << sum(arr  ) << std::endl;
    std::cout << sum(arr_h) << std::endl;
    std::cout << sum(arr_d) << std::endl;
    Array<float[10],yakl::PoolSpace> arr_s("arr_s");
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
    CSArray<float,3,2> csarray;
    csarray = 2;
    csarray(2,1) = 1;
    std::cout << csarray;
    std::cout << csarray.extents();
    std::cout << csarray.lbounds();
    std::cout << csarray.ubounds();
    std::cout << sum(csarray) << std::endl;
    FSArray<float,{1,3},{1,2}> fsarray;
    fsarray = 2;
    fsarray(3,2) = 1;
    std::cout << fsarray;
    std::cout << fsarray.extents();
    std::cout << fsarray.lbounds();
    std::cout << fsarray.ubounds();
    std::cout << sum(fsarray) << std::endl;
    Array<float ***,yakl::PoolSpace,FStyle> farr("arr",{1,3},{1,3},{1,3});
    farr = 3;
    Kokkos::parallel_for( "" , 1 , KOKKOS_LAMBDA (int i) { farr(1,1,1) = 1; });
    Kokkos::fence();
    std::cout << farr.lbounds();
    std::cout << farr.ubounds();
    auto farr_re = farr.reshape(27);
    std::cout << farr_re.is_fstyle << " , " << farr_re.rank << std::endl;
    std::cout << farr_re.lbounds();
    std::cout << farr_re.ubounds();
    std::cout << sum(farr) << std::endl;
    Kokkos::parallel_for( "" , 1 , KOKKOS_LAMBDA (int i) { farr_re.Array::operator()(27) = 1; });
    Kokkos::fence();
    std::cout << farr_re;
    std::cout << sum(farr_re) << std::endl;
    Array<float *,yakl::PoolSpace,FStyle> farr2("farr2",{1,10});
    Kokkos::parallel_for( "mykernel" , Kokkos::RangePolicy(1,11) , KOKKOS_LAMBDA (int i) {
      farr2(i) = i;
    });
    std::cout << farr2;
    std::cout << farr2.reshape(5,2).slice<1>(COLON,2);
    Bounds<3,FStyle,false> bnds({1,3},{-1,5},{2,7,2});
  }
  yakl::finalize();
  Kokkos::finalize(); 
  return 0;
}

