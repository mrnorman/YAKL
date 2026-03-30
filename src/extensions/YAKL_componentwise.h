
#pragma once

namespace yakl {
  namespace componentwise {

    template <class V1, class V2, class F> requires std::is_arithmetic_v<V1> && std::is_arithmetic_v<V2>
    inline auto binary( V1 const & l , V2 const & r , F const & f ) {
      return f(l,r);
    }
    template <class V1, class V2, class F> requires yakl::is_SArray<V1> && std::is_arithmetic_v<V2>
    inline auto binary( V1 const & l , V2 const & r , F const & f ) {
      typename V1::template TypeAs<decltype(f(l.data()[0],r))> ret;
      for (int i=0; i < l.size(); i++) { ret.data()[i] = f(l.data()[i],r); }
      return ret;
    }
    template <class V1, class V2, class F> requires std::is_arithmetic_v<V1> && yakl::is_SArray<V2>
    inline auto binary( V1 const & l , V2 const & r , F const & f ) {
      typename V2::template TypeAs<decltype(f(l,r.data()[0]))> ret;
      for (int i=0; i < r.size(); i++) { ret.data()[i] = f(l,r.data()[i]); }
      return ret;
    }
    template <class V1, class V2, class F> requires yakl::is_SArray<V1> && yakl::is_SArray<V2>
    inline auto binary( V1 const & l , V2 const & r , F const & f ) {
      typename V1::template TypeAs<decltype(f(l.data()[0],r.data()[0]))> ret;
      for (int i=0; i < l.size(); i++) { ret.data()[i] = f(l.data()[i],r.data()[i]); }
      return ret;
    }
    template <class V1, class V2, class F> requires yakl::is_Array<V1> && std::is_arithmetic_v<V2>
    inline auto binary( V1 const & l , V2 const & r , F const & f ) {
      auto ret = l.template clone_object<typename V1::memory_space,decltype(f(l.data()[0],r))>();
      Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                            Kokkos::RangePolicy<typename V1::execution_space>(0,l.size()) ,
                            KOKKOS_LAMBDA (int i) {
        ret.data()[i] = f(l.data()[i],r);
      } );
      return ret;
    }
    template <class V1, class V2, class F> requires std::is_arithmetic_v<V1> && yakl::is_Array<V2>
    inline auto binary( V1 const & l , V2 const & r , F const & f ) {
      auto ret = r.template clone_object<typename V2::memory_space,decltype(f(l,r.data()[0]))>();
      Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                            Kokkos::RangePolicy<typename V2::execution_space>(0,r.size()) ,
                            KOKKOS_LAMBDA (int i) {
        ret.data()[i] = f(l,r.data()[i]);
      } );
      return ret;
    }
    template <class V1, class V2, class F> requires yakl::is_Array<V1> && yakl::is_Array<V2>
    inline auto binary( V1 const & l , V2 const & r , F const & f ) {
      auto ret = l.template clone_object<typename V1::memory_space,decltype(f(l.data()[0],r.data()[0]))>();
      Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                            Kokkos::RangePolicy<typename V1::execution_space>(0,l.size()) ,
                            KOKKOS_LAMBDA (int i) {
        ret.data()[i] = f(l.data()[i],r.data()[i]);
      } );
      return ret;
    }


    template <class V1, class V2> auto operator+( V1 const & l , V2 const & r ) {
      return binary( l , r , []<class T1,class T2>(T1 l,T2 r) constexpr {return l+r;} );
    }

    template <class V1, class V2> auto operator-( V1 const & l , V2 const & r ) {
      return binary( l , r , []<class T1,class T2>(T1 l,T2 r) constexpr {return l-r;} );
    }

    template <class V1, class V2> auto operator*( V1 const & l , V2 const & r ) {
      return binary( l , r , []<class T1,class T2>(T1 l,T2 r) constexpr {return l*r;} );
    }

    template <class V1, class V2> auto operator/( V1 const & l , V2 const & r ) {
      return binary( l , r , []<class T1,class T2>(T1 l,T2 r) constexpr {return l/r;} );
    }

    template <class V1, class V2> auto operator<( V1 const & l , V2 const & r ) {
      return binary( l , r , []<class T1,class T2>(T1 l,T2 r) constexpr {return l<r;} );
    }

    template <class V1, class V2> auto operator>( V1 const & l , V2 const & r ) {
      return binary( l , r , []<class T1,class T2>(T1 l,T2 r) constexpr {return l>r;} );
    }

    template <class V1, class V2> auto operator<=( V1 const & l , V2 const & r ) {
      return binary( l , r , []<class T1,class T2>(T1 l,T2 r) constexpr {return l<=r;} );
    }

    template <class V1, class V2> auto operator>=( V1 const & l , V2 const & r ) {
      return binary( l , r , []<class T1,class T2>(T1 l,T2 r) constexpr {return l>=r;} );
    }

    template <class V1, class V2> auto operator==( V1 const & l , V2 const & r ) {
      return binary( l , r , []<class T1,class T2>(T1 l,T2 r) constexpr {return l==r;} );
    }

    template <class V1, class V2> auto operator!=( V1 const & l , V2 const & r ) {
      return binary( l , r , []<class T1,class T2>(T1 l,T2 r) constexpr {return l!=r;} );
    }

    template <class V1, class V2> auto operator&&( V1 const & l , V2 const & r ) {
      return binary( l , r , []<class T1,class T2>(T1 l,T2 r) constexpr {return l&&r;} );
    }

    template <class V1, class V2> auto operator||( V1 const & l , V2 const & r ) {
      return binary( l , r , []<class T1,class T2>(T1 l,T2 r) constexpr {return l||r;} );
    }




    template <class V, class F> requires std::is_arithmetic_v<V>
    inline auto unary( V const & v , F const & f ) {
      return f(v);
    }
    template <class V, class F> requires yakl::is_SArray<V>
    inline auto unary( V const & v , F const & f ) {
      typename V::template TypeAs<decltype(f(v.data()[0]))> ret;
      for (int i=0; i < v.size(); i++) { ret.data()[i] = f(v.data()[i]); }
      return ret;
    }
    template <class V, class F> requires yakl::is_Array<V>
    inline auto unary( V const & v , F const & f ) {
      auto ret = v.template clone_object<typename V::memory_space,decltype(f(v.data()[0]))>();
      Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                            Kokkos::RangePolicy<typename V::execution_space>(0,v.size()) ,
                            KOKKOS_LAMBDA (int i) {
        ret.data()[i] = f(v.data()[i]);
      } );
      return ret;
    }

    template <class V> auto operator!( V const & v ) {
      return unary( v , []<class T>(T v) constexpr {return !v;} );
    }

    template <class V> auto operator+( V const & v ) {
      return unary( v , []<class T>(T v) constexpr {return +v;} );
    }

    template <class V> auto operator-( V const & v ) {
      return unary( v , []<class T>(T v) constexpr {return -v;} );
    }

    template <class V> auto abs( V const & v ) {
      return unary( v , []<class T>(T v) constexpr {return std::abs(v);} );
    }

    template <class V> auto sqrt( V const & v ) {
      return unary( v , []<class T>(T v) constexpr {return std::sqrt(v);} );
    }

    template <class V> auto cbrt( V const & v ) {
      return unary( v , []<class T>(T v) constexpr {return std::cbrt(v);} );
    }

    template <class V, class V2> auto pow( V const & v , V2 const & v2 ) requires std::is_arithmetic_v<V2> {
      using type = typename V::value_type;
      return unary( v , [=]<class T>(T v) constexpr {return std::pow(v,static_cast<type>(v2));} );
    }

    template <class V> auto sin( V const & v ) {
      return unary( v , []<class T>(T v) constexpr {return std::sin(v);} );
    }

    template <class V> auto cos( V const & v ) {
      return unary( v , []<class T>(T v) constexpr {return std::cos(v);} );
    }

    template <class V> auto tan( V const & v ) {
      return unary( v , []<class T>(T v) constexpr {return std::tan(v);} );
    }

    template <class V> auto asin( V const & v ) {
      return unary( v , []<class T>(T v) constexpr {return std::asin(v);} );
    }

    template <class V> auto acos( V const & v ) {
      return unary( v , []<class T>(T v) constexpr {return std::acos(v);} );
    }

    template <class V> auto atan( V const & v ) {
      return unary( v , []<class T>(T v) constexpr {return std::atan(v);} );
    }

    template <class V> auto exp( V const & v ) {
      return unary( v , []<class T>(T v) constexpr {return std::exp(v);} );
    }

    template <class V> auto log( V const & v ) {
      return unary( v , []<class T>(T v) constexpr {return std::log(v);} );
    }

    template <class V> auto log10( V const & v ) {
      return unary( v , []<class T>(T v) constexpr {return std::log10(v);} );
    }

    template <class V> auto log2( V const & v ) {
      return unary( v , []<class T>(T v) constexpr {return std::log2(v);} );
    }

    template <class V> auto floor( V const & v ) {
      return unary( v , []<class T>(T v) constexpr {return std::floor(v);} );
    }

    template <class V> auto ceil( V const & v ) {
      return unary( v , []<class T>(T v) constexpr {return std::ceil(v);} );
    }

    template <class V> auto round( V const & v ) {
      return unary( v , []<class T>(T v) constexpr {return std::round(v);} );
    }

    template <class V> auto isnan( V const & v ) {
      return unary( v , []<class T>(T v) constexpr {return std::isnan(v);} );
    }

    template <class V> auto isinf( V const & v ) {
      return unary( v , []<class T>(T v) constexpr {return std::isinf(v);} );
    }

  }
}
