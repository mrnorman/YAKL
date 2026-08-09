
#pragma once

namespace yakl {
  namespace componentwise {

    template <class V1, class V2>
    bool constexpr has_array_operand = yakl::is_Array<V1> || yakl::is_Array<V2>;

    template <class V1, class V2>
    bool constexpr compatible_array_operands = V1::is_cstyle == V2::is_cstyle &&
                                               std::is_same_v<typename V1::memory_space,typename V2::memory_space>;

    template <class V1, class V2>
    bool constexpr compatible_stack_operands = V1::is_cstyle == V2::is_cstyle;

    template <class V1, class V2>
    KOKKOS_INLINE_FUNCTION bool same_shape(V1 const & a, V2 const & b) {
      if constexpr (V1::rank() != V2::rank()) {
        return false;
      } else {
        for (int i=0; i < V1::rank(); i++) {
          if (a.extent(i) != b.extent(i)) return false;
        }
        return true;
      }
    }

    template <class V1, class V2, class F> requires std::is_arithmetic_v<V1> && std::is_arithmetic_v<V2>
    KOKKOS_INLINE_FUNCTION auto binary( V1 const & l , V2 const & r , F const & f ) {
      return f(l,r);
    }
    template <class V1, class V2, class F> requires yakl::is_SArray<V1> && std::is_arithmetic_v<V2>
    KOKKOS_INLINE_FUNCTION auto binary( V1 const & l , V2 const & r , F const & f ) {
      typename V1::template TypeAs<decltype(f(l.data()[0],r))> ret;
      for (int i=0; i < l.size(); i++) { ret.data()[i] = f(l.data()[i],r); }
      return ret;
    }
    template <class V1, class V2, class F> requires std::is_arithmetic_v<V1> && yakl::is_SArray<V2>
    KOKKOS_INLINE_FUNCTION auto binary( V1 const & l , V2 const & r , F const & f ) {
      typename V2::template TypeAs<decltype(f(l,r.data()[0]))> ret;
      for (int i=0; i < r.size(); i++) { ret.data()[i] = f(l,r.data()[i]); }
      return ret;
    }
    template <class V1, class V2, class F> requires yakl::is_SArray<V1> && yakl::is_SArray<V2>
    KOKKOS_INLINE_FUNCTION auto binary( V1 const & l , V2 const & r , F const & f ) {
      static_assert(compatible_stack_operands<V1,V2>,
                    "ERROR: componentwise binary operation cannot mix SArray and SArray_F operands");
      static_assert(V1::rank == V2::rank,"ERROR: componentwise binary operation requires equal ranks");
      static_assert(V1::size() == V2::size(),"ERROR: componentwise binary operation requires equal sizes");
      if constexpr (kokkos_debug) {
        for (int i=0; i < V1::rank; i++) {
          if (l.extent(i) != r.extent(i)) {
            Kokkos::abort("ERROR: componentwise binary operation requires identical shapes");
          }
        }
      }
      typename V1::template TypeAs<decltype(f(l.data()[0],r.data()[0]))> ret;
      for (int i=0; i < l.size(); i++) { ret.data()[i] = f(l.data()[i],r.data()[i]); }
      return ret;
    }
    template <class V1, class V2, class F> requires yakl::is_Array<V1> && std::is_arithmetic_v<V2>
    inline auto binary( V1 const & l , V2 const & r , F const & f ) ->
    decltype(l.template clone_object<typename V1::memory_space,decltype(f(l.data()[0],r))>())
    {
      if constexpr (kokkos_debug) {
        if (!l.is_allocated()) Kokkos::abort("ERROR: componentwise binary operation on an unallocated Array");
      }
      auto ret = l.template clone_object<typename V1::memory_space,decltype(f(l.data()[0],r))>();
      if constexpr (yakl_auto_profile) timer_start("yakl::componentwise::binary");
      Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                            Kokkos::RangePolicy<typename V1::execution_space,Kokkos::IndexType<uindex_t>>(
                              0,checked_uindex(l.size(),"ERROR: Array size exceeds the configured index range")) ,
                            KOKKOS_LAMBDA (uindex_t i) {
        auto &lloc = l;
        auto &rloc = r;
        ret.data()[i] = f(lloc.data()[i],rloc);
      } );
      if constexpr (yakl_auto_profile) timer_stop("yakl::componentwise::binary");
      if constexpr (yakl_auto_fence) Kokkos::fence();
      return ret;
    }
    template <class V1, class V2, class F> requires std::is_arithmetic_v<V1> && yakl::is_Array<V2>
    inline auto binary( V1 const & l , V2 const & r , F const & f ) ->
    decltype(r.template clone_object<typename V2::memory_space,decltype(f(l,r.data()[0]))>())
    {
      if constexpr (kokkos_debug) {
        if (!r.is_allocated()) Kokkos::abort("ERROR: componentwise binary operation on an unallocated Array");
      }
      auto ret = r.template clone_object<typename V2::memory_space,decltype(f(l,r.data()[0]))>();
      if constexpr (yakl_auto_profile) timer_start("yakl::componentwise::binary");
      Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                            Kokkos::RangePolicy<typename V2::execution_space,Kokkos::IndexType<uindex_t>>(
                              0,checked_uindex(r.size(),"ERROR: Array size exceeds the configured index range")) ,
                            KOKKOS_LAMBDA (uindex_t i) {
        auto &lloc = l;
        auto &rloc = r;
        ret.data()[i] = f(lloc,rloc.data()[i]);
      } );
      if constexpr (yakl_auto_profile) timer_stop("yakl::componentwise::binary");
      if constexpr (yakl_auto_fence) Kokkos::fence();
      return ret;
    }
    template <class V1, class V2, class F> requires yakl::is_Array<V1> && yakl::is_Array<V2>
    inline auto binary( V1 const & l , V2 const & r , F const & f ) ->
    decltype(l.template clone_object<typename V1::memory_space,decltype(f(l.data()[0],r.data()[0]))>())
    {
      static_assert(compatible_array_operands<V1,V2>,
                    "ERROR: componentwise binary operation requires the same Array style and memory space");
      if constexpr (kokkos_debug) {
        if (!l.is_allocated() || !r.is_allocated()) {
          Kokkos::abort("ERROR: componentwise binary operation on an unallocated Array");
        }
        if (!same_shape(l,r)) {
          Kokkos::abort("ERROR: componentwise binary operation requires arrays with identical shapes");
        }
      }
      auto ret = l.template clone_object<typename V1::memory_space,decltype(f(l.data()[0],r.data()[0]))>();
      if constexpr (yakl_auto_profile) timer_start("yakl::componentwise::binary");
      Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                            Kokkos::RangePolicy<typename V1::execution_space,Kokkos::IndexType<uindex_t>>(
                              0,checked_uindex(l.size(),"ERROR: Array size exceeds the configured index range")) ,
                            KOKKOS_LAMBDA (uindex_t i) {
        auto &lloc = l;
        auto &rloc = r;
        ret.data()[i] = f(lloc.data()[i],rloc.data()[i]);
      } );
      if constexpr (yakl_auto_profile) timer_stop("yakl::componentwise::binary");
      if constexpr (yakl_auto_fence) Kokkos::fence();
      return ret;
    }


    struct AddOp{template <class V1,class V2> requires std::is_arithmetic_v<V1> && std::is_arithmetic_v<V2> KOKKOS_INLINE_FUNCTION auto operator()(V1 l,V2 r)const{return l+r;} };
    template <class V1, class V2> requires (!has_array_operand<V1,V2>)
    KOKKOS_INLINE_FUNCTION auto operator+( V1 const & l , V2 const & r ) {
      return binary( l , r , AddOp{} );
    }
    template <class V1, class V2> requires has_array_operand<V1,V2>
    inline auto operator+( V1 const & l , V2 const & r ) {
      return binary( l , r , AddOp{} );
    }

    struct SubOp{template <class V1,class V2> requires std::is_arithmetic_v<V1> && std::is_arithmetic_v<V2> KOKKOS_INLINE_FUNCTION auto operator()(V1 l,V2 r)const{return l-r;} };
    template <class V1, class V2> requires (!has_array_operand<V1,V2>)
    KOKKOS_INLINE_FUNCTION auto operator-( V1 const & l , V2 const & r ) {
      return binary( l , r , SubOp{} );
    }
    template <class V1, class V2> requires has_array_operand<V1,V2>
    inline auto operator-( V1 const & l , V2 const & r ) {
      return binary( l , r , SubOp{} );
    }

    struct MultOp{template <class V1,class V2> requires std::is_arithmetic_v<V1> && std::is_arithmetic_v<V2> KOKKOS_INLINE_FUNCTION auto operator()(V1 l,V2 r)const{return l*r;} };
    template <class V1, class V2> requires (!has_array_operand<V1,V2>)
    KOKKOS_INLINE_FUNCTION auto operator*( V1 const & l , V2 const & r ) {
      return binary( l , r , MultOp{} );
    }
    template <class V1, class V2> requires has_array_operand<V1,V2>
    inline auto operator*( V1 const & l , V2 const & r ) {
      return binary( l , r , MultOp{} );
    }

    struct DivOp{template <class V1,class V2> requires std::is_arithmetic_v<V1> && std::is_arithmetic_v<V2> KOKKOS_INLINE_FUNCTION auto operator()(V1 l,V2 r)const{return l/r;} };
    template <class V1, class V2> requires (!has_array_operand<V1,V2>)
    KOKKOS_INLINE_FUNCTION auto operator/( V1 const & l , V2 const & r ) {
      return binary( l , r , DivOp{} );
    }
    template <class V1, class V2> requires has_array_operand<V1,V2>
    inline auto operator/( V1 const & l , V2 const & r ) {
      return binary( l , r , DivOp{} );
    }

    struct LTOp{template <class V1,class V2> requires std::is_arithmetic_v<V1> && std::is_arithmetic_v<V2> KOKKOS_INLINE_FUNCTION auto operator()(V1 l,V2 r)const{return l<r;} };
    template <class V1, class V2> requires (!has_array_operand<V1,V2>)
    KOKKOS_INLINE_FUNCTION auto operator<( V1 const & l , V2 const & r ) {
      return binary( l , r , LTOp{} );
    }
    template <class V1, class V2> requires has_array_operand<V1,V2>
    inline auto operator<( V1 const & l , V2 const & r ) {
      return binary( l , r , LTOp{} );
    }

    struct GTOp{template <class V1,class V2> requires std::is_arithmetic_v<V1> && std::is_arithmetic_v<V2> KOKKOS_INLINE_FUNCTION auto operator()(V1 l,V2 r)const{return l>r;} };
    template <class V1, class V2> requires (!has_array_operand<V1,V2>)
    KOKKOS_INLINE_FUNCTION auto operator>( V1 const & l , V2 const & r ) {
      return binary( l , r , GTOp{} );
    }
    template <class V1, class V2> requires has_array_operand<V1,V2>
    inline auto operator>( V1 const & l , V2 const & r ) {
      return binary( l , r , GTOp{} );
    }

    struct LEOp{template <class V1,class V2> requires std::is_arithmetic_v<V1> && std::is_arithmetic_v<V2> KOKKOS_INLINE_FUNCTION auto operator()(V1 l,V2 r)const{return l<=r;} };
    template <class V1, class V2> requires (!has_array_operand<V1,V2>)
    KOKKOS_INLINE_FUNCTION auto operator<=( V1 const & l , V2 const & r ) {
      return binary( l , r , LEOp{} );
    }
    template <class V1, class V2> requires has_array_operand<V1,V2>
    inline auto operator<=( V1 const & l , V2 const & r ) {
      return binary( l , r , LEOp{} );
    }

    struct GEOp{template <class V1,class V2> requires std::is_arithmetic_v<V1> && std::is_arithmetic_v<V2> KOKKOS_INLINE_FUNCTION auto operator()(V1 l,V2 r)const{return l>=r;} };
    template <class V1, class V2> requires (!has_array_operand<V1,V2>)
    KOKKOS_INLINE_FUNCTION auto operator>=( V1 const & l , V2 const & r ) {
      return binary( l , r , GEOp{} );
    }
    template <class V1, class V2> requires has_array_operand<V1,V2>
    inline auto operator>=( V1 const & l , V2 const & r ) {
      return binary( l , r , GEOp{} );
    }

    struct EEOp{template <class V1,class V2> requires std::is_arithmetic_v<V1> && std::is_arithmetic_v<V2> KOKKOS_INLINE_FUNCTION auto operator()(V1 l,V2 r)const{return l==r;} };
    template <class V1, class V2> requires (!has_array_operand<V1,V2>)
    KOKKOS_INLINE_FUNCTION auto operator==( V1 const & l , V2 const & r ) {
      return binary( l , r , EEOp{} );
    }
    template <class V1, class V2> requires has_array_operand<V1,V2>
    inline auto operator==( V1 const & l , V2 const & r ) {
      return binary( l , r , EEOp{} );
    }

    struct NEOp{template <class V1,class V2> requires std::is_arithmetic_v<V1> && std::is_arithmetic_v<V2> KOKKOS_INLINE_FUNCTION auto operator()(V1 l,V2 r)const{return l!=r;} };
    template <class V1, class V2> requires (!has_array_operand<V1,V2>)
    KOKKOS_INLINE_FUNCTION auto operator!=( V1 const & l , V2 const & r ) {
      return binary( l , r , NEOp{} );
    }
    template <class V1, class V2> requires has_array_operand<V1,V2>
    inline auto operator!=( V1 const & l , V2 const & r ) {
      return binary( l , r , NEOp{} );
    }

    struct AndOp{template <class V1,class V2> requires std::is_arithmetic_v<V1> && std::is_arithmetic_v<V2> KOKKOS_INLINE_FUNCTION auto operator()(V1 l,V2 r)const{return l&&r;} };
    template <class V1, class V2> requires (!has_array_operand<V1,V2>)
    KOKKOS_INLINE_FUNCTION auto operator&&( V1 const & l , V2 const & r ) {
      return binary( l , r , AndOp{} );
    }
    template <class V1, class V2> requires has_array_operand<V1,V2>
    inline auto operator&&( V1 const & l , V2 const & r ) {
      return binary( l , r , AndOp{} );
    }

    struct OrOp{template <class V1,class V2> requires std::is_arithmetic_v<V1> && std::is_arithmetic_v<V2> KOKKOS_INLINE_FUNCTION auto operator()(V1 l,V2 r)const{return l||r;} };
    template <class V1, class V2> requires (!has_array_operand<V1,V2>)
    KOKKOS_INLINE_FUNCTION auto operator||( V1 const & l , V2 const & r ) {
      return binary( l , r , OrOp{} );
    }
    template <class V1, class V2> requires has_array_operand<V1,V2>
    inline auto operator||( V1 const & l , V2 const & r ) {
      return binary( l , r , OrOp{} );
    }




    template <class V, class F> requires std::is_arithmetic_v<V>
    KOKKOS_INLINE_FUNCTION auto unary( V const & v , F const & f ) {
      return f(v);
    }
    template <class V, class F> requires yakl::is_SArray<V>
    KOKKOS_INLINE_FUNCTION auto unary( V const & v , F const & f ) {
      typename V::template TypeAs<decltype(f(v.data()[0]))> ret;
      for (int i=0; i < v.size(); i++) { ret.data()[i] = f(v.data()[i]); }
      return ret;
    }
    template <class V, class F> requires yakl::is_Array<V>
    inline auto unary( V const & v , F const & f ) ->
    decltype(v.template clone_object<typename V::memory_space,decltype(f(v.data()[0]))>())
    {
      if constexpr (kokkos_debug) {
        if (!v.is_allocated()) Kokkos::abort("ERROR: componentwise unary operation on an unallocated Array");
      }
      auto ret = v.template clone_object<typename V::memory_space,decltype(f(v.data()[0]))>();
      if constexpr (yakl_auto_profile) timer_start("yakl::componentwise::unary");
      Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                            Kokkos::RangePolicy<typename V::execution_space,Kokkos::IndexType<uindex_t>>(
                              0,checked_uindex(v.size(),"ERROR: Array size exceeds the configured index range")) ,
                            KOKKOS_LAMBDA (uindex_t i) {
        ret.data()[i] = f(v.data()[i]);
      } );
      if constexpr (yakl_auto_profile) timer_stop("yakl::componentwise::unary");
      if constexpr (yakl_auto_fence) Kokkos::fence();
      return ret;
    }

    struct NotOp{template <class V> requires std::is_arithmetic_v<V> KOKKOS_INLINE_FUNCTION auto operator()(V v)const{return !v;} };
    template <class V> requires (!yakl::is_Array<V>)
    KOKKOS_INLINE_FUNCTION auto operator!( V const & v ) {
      return unary( v , NotOp{} );
    }
    template <class V> requires yakl::is_Array<V>
    inline auto operator!( V const & v ) {
      return unary( v , NotOp{} );
    }

    struct PosOp{template <class V> requires std::is_arithmetic_v<V> KOKKOS_INLINE_FUNCTION auto operator()(V v)const{return +v;} };
    template <class V> requires (!yakl::is_Array<V>)
    KOKKOS_INLINE_FUNCTION auto operator+( V const & v ) {
      return unary( v , PosOp{} );
    }
    template <class V> requires yakl::is_Array<V>
    inline auto operator+( V const & v ) {
      return unary( v , PosOp{} );
    }

    struct NegOp{template <class V> requires std::is_arithmetic_v<V> KOKKOS_INLINE_FUNCTION auto operator()(V v)const{return -v;} };
    template <class V> requires (!yakl::is_Array<V>)
    KOKKOS_INLINE_FUNCTION auto operator-( V const & v ) {
      return unary( v , NegOp{} );
    }
    template <class V> requires yakl::is_Array<V>
    inline auto operator-( V const & v ) {
      return unary( v , NegOp{} );
    }

    struct AbsOp{template <class V> requires std::is_arithmetic_v<V> KOKKOS_INLINE_FUNCTION auto operator()(V v)const{return std::abs(v);} };
    template <class V> requires (!yakl::is_Array<V>)
    KOKKOS_INLINE_FUNCTION auto abs( V const & v ) {
      return unary( v , AbsOp{} );
    }
    template <class V> requires yakl::is_Array<V>
    inline auto abs( V const & v ) {
      return unary( v , AbsOp{} );
    }

    struct SqrtOp{template <class V> requires std::is_arithmetic_v<V> KOKKOS_INLINE_FUNCTION auto operator()(V v)const{return std::sqrt(v);} };
    template <class V> requires (!yakl::is_Array<V>)
    KOKKOS_INLINE_FUNCTION auto sqrt( V const & v ) {
      return unary( v , SqrtOp{} );
    }
    template <class V> requires yakl::is_Array<V>
    inline auto sqrt( V const & v ) {
      return unary( v , SqrtOp{} );
    }

    struct CbrtOp{template <class V> requires std::is_arithmetic_v<V> KOKKOS_INLINE_FUNCTION auto operator()(V v)const{return std::cbrt(v);} };
    template <class V> requires (!yakl::is_Array<V>)
    KOKKOS_INLINE_FUNCTION auto cbrt( V const & v ) {
      return unary( v , CbrtOp{} );
    }
    template <class V> requires yakl::is_Array<V>
    inline auto cbrt( V const & v ) {
      return unary( v , CbrtOp{} );
    }

    template <class V2> requires std::is_arithmetic_v<V2>
    struct PowOp{
      V2 v2;
      template <class V> requires std::is_arithmetic_v<V>
      KOKKOS_INLINE_FUNCTION auto operator()(V v)const{return std::pow(v,v2);}
    };
    template <class V, class V2> requires std::is_arithmetic_v<V2> && (!yakl::is_Array<V>)
    KOKKOS_INLINE_FUNCTION auto pow( V const & v , V2 const & v2 ) {
      return unary( v , PowOp{v2} );
    }
    template <class V, class V2> requires std::is_arithmetic_v<V2> && yakl::is_Array<V>
    inline auto pow( V const & v , V2 const & v2 ) {
      return unary( v , PowOp{v2} );
    }

    struct SinOp{template <class V> requires std::is_arithmetic_v<V> KOKKOS_INLINE_FUNCTION auto operator()(V v)const{return std::sin(v);} };
    template <class V> requires (!yakl::is_Array<V>)
    KOKKOS_INLINE_FUNCTION auto sin( V const & v ) {
      return unary( v , SinOp{} );
    }
    template <class V> requires yakl::is_Array<V>
    inline auto sin( V const & v ) {
      return unary( v , SinOp{} );
    }

    struct CosOp{template <class V> requires std::is_arithmetic_v<V> KOKKOS_INLINE_FUNCTION auto operator()(V v)const{return std::cos(v);} };
    template <class V> requires (!yakl::is_Array<V>)
    KOKKOS_INLINE_FUNCTION auto cos( V const & v ) {
      return unary( v , CosOp{} );
    }
    template <class V> requires yakl::is_Array<V>
    inline auto cos( V const & v ) {
      return unary( v , CosOp{} );
    }

    struct TanOp{template <class V> requires std::is_arithmetic_v<V> KOKKOS_INLINE_FUNCTION auto operator()(V v)const{return std::tan(v);} };
    template <class V> requires (!yakl::is_Array<V>)
    KOKKOS_INLINE_FUNCTION auto tan( V const & v ) {
      return unary( v , TanOp{} );
    }
    template <class V> requires yakl::is_Array<V>
    inline auto tan( V const & v ) {
      return unary( v , TanOp{} );
    }

    struct AsinOp{template <class V> requires std::is_arithmetic_v<V> KOKKOS_INLINE_FUNCTION auto operator()(V v)const{return std::asin(v);} };
    template <class V> requires (!yakl::is_Array<V>)
    KOKKOS_INLINE_FUNCTION auto asin( V const & v ) {
      return unary( v , AsinOp{} );
    }
    template <class V> requires yakl::is_Array<V>
    inline auto asin( V const & v ) {
      return unary( v , AsinOp{} );
    }

    struct AcosOp{template <class V> requires std::is_arithmetic_v<V> KOKKOS_INLINE_FUNCTION auto operator()(V v)const{return std::acos(v);} };
    template <class V> requires (!yakl::is_Array<V>)
    KOKKOS_INLINE_FUNCTION auto acos( V const & v ) {
      return unary( v , AcosOp{} );
    }
    template <class V> requires yakl::is_Array<V>
    inline auto acos( V const & v ) {
      return unary( v , AcosOp{} );
    }

    struct AtanOp{template <class V> requires std::is_arithmetic_v<V> KOKKOS_INLINE_FUNCTION auto operator()(V v)const{return std::atan(v);} };
    template <class V> requires (!yakl::is_Array<V>)
    KOKKOS_INLINE_FUNCTION auto atan( V const & v ) {
      return unary( v , AtanOp{} );
    }
    template <class V> requires yakl::is_Array<V>
    inline auto atan( V const & v ) {
      return unary( v , AtanOp{} );
    }

    struct ExpOp{template <class V> requires std::is_arithmetic_v<V> KOKKOS_INLINE_FUNCTION auto operator()(V v)const{return std::exp(v);} };
    template <class V> requires (!yakl::is_Array<V>)
    KOKKOS_INLINE_FUNCTION auto exp( V const & v ) {
      return unary( v , ExpOp{} );
    }
    template <class V> requires yakl::is_Array<V>
    inline auto exp( V const & v ) {
      return unary( v , ExpOp{} );
    }

    struct LogOp{template <class V> requires std::is_arithmetic_v<V> KOKKOS_INLINE_FUNCTION auto operator()(V v)const{return std::log(v);} };
    template <class V> requires (!yakl::is_Array<V>)
    KOKKOS_INLINE_FUNCTION auto log( V const & v ) {
      return unary( v , LogOp{} );
    }
    template <class V> requires yakl::is_Array<V>
    inline auto log( V const & v ) {
      return unary( v , LogOp{} );
    }

    struct Log10Op{template <class V> requires std::is_arithmetic_v<V> KOKKOS_INLINE_FUNCTION auto operator()(V v)const{return std::log10(v);} };
    template <class V> requires (!yakl::is_Array<V>)
    KOKKOS_INLINE_FUNCTION auto log10( V const & v ) {
      return unary( v , Log10Op{} );
    }
    template <class V> requires yakl::is_Array<V>
    inline auto log10( V const & v ) {
      return unary( v , Log10Op{} );
    }

    struct Log2Op{template <class V> requires std::is_arithmetic_v<V> KOKKOS_INLINE_FUNCTION auto operator()(V v)const{return std::log2(v);} };
    template <class V> requires (!yakl::is_Array<V>)
    KOKKOS_INLINE_FUNCTION auto log2( V const & v ) {
      return unary( v , Log2Op{} );
    }
    template <class V> requires yakl::is_Array<V>
    inline auto log2( V const & v ) {
      return unary( v , Log2Op{} );
    }

    struct FloorOp{template <class V> requires std::is_arithmetic_v<V> KOKKOS_INLINE_FUNCTION auto operator()(V v)const{return std::floor(v);} };
    template <class V> requires (!yakl::is_Array<V>)
    KOKKOS_INLINE_FUNCTION auto floor( V const & v ) {
      return unary( v , FloorOp{} );
    }
    template <class V> requires yakl::is_Array<V>
    inline auto floor( V const & v ) {
      return unary( v , FloorOp{} );
    }

    struct CeilOp{template <class V> requires std::is_arithmetic_v<V> KOKKOS_INLINE_FUNCTION auto operator()(V v)const{return std::ceil(v);} };
    template <class V> requires (!yakl::is_Array<V>)
    KOKKOS_INLINE_FUNCTION auto ceil( V const & v ) {
      return unary( v , CeilOp{} );
    }
    template <class V> requires yakl::is_Array<V>
    inline auto ceil( V const & v ) {
      return unary( v , CeilOp{} );
    }

    struct RoundOp{template <class V> requires std::is_arithmetic_v<V> KOKKOS_INLINE_FUNCTION auto operator()(V v)const{return std::round(v);} };
    template <class V> requires (!yakl::is_Array<V>)
    KOKKOS_INLINE_FUNCTION auto round( V const & v ) {
      return unary( v , RoundOp{} );
    }
    template <class V> requires yakl::is_Array<V>
    inline auto round( V const & v ) {
      return unary( v , RoundOp{} );
    }

    struct IsnanOp{template <class V> requires std::is_arithmetic_v<V> KOKKOS_INLINE_FUNCTION auto operator()(V v)const{return std::isnan(v);} };
    template <class V> requires (!yakl::is_Array<V>)
    KOKKOS_INLINE_FUNCTION auto isnan( V const & v ) {
      return unary( v , IsnanOp{} );
    }
    template <class V> requires yakl::is_Array<V>
    inline auto isnan( V const & v ) {
      return unary( v , IsnanOp{} );
    }

    struct IsinfOp{template <class V> requires std::is_arithmetic_v<V> KOKKOS_INLINE_FUNCTION auto operator()(V v)const{return std::isinf(v);} };
    template <class V> requires (!yakl::is_Array<V>)
    KOKKOS_INLINE_FUNCTION auto isinf( V const & v ) {
      return unary( v , IsinfOp{} );
    }
    template <class V> requires yakl::is_Array<V>
    inline auto isinf( V const & v ) {
      return unary( v , IsinfOp{} );
    }

  }
}
