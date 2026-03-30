
#pragma once
// Included by YAKL.h

namespace yakl {
  namespace intrinsics {

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // COMPONENTWISE
    //////////////////////////////////////////////////////////////////////////////////////////////////////

    // ABS
    template <class ViewType> inline auto abs(ViewType const & in) {
      if constexpr (kokkos_debug) if (!in.span_is_contiguous()) Kokkos::abort("ERROR: abs on non-contiguous View");
      if constexpr (kokkos_debug) if (!in.is_allocated      ()) Kokkos::abort("ERROR: abs on unallocated View");
      if constexpr (is_SArray<ViewType>) {
        ViewType ret;
        for (int i=0; i < in.size(); i++) { ret.data()[i] = std::abs(in.data()[i]); }
        return ret;
      } else {
        auto ret = in.clone_object();
        Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                              Kokkos::RangePolicy<typename ViewType::execution_space>(0,in.size()) ,
                              KOKKOS_LAMBDA (int i) {
          ret.data()[i] = std::abs(in.data()[i]);
        } );
        return ret;
      }
    }


    // SIGN
    template <class ViewType>
    KOKKOS_INLINE_FUNCTION auto sign( ViewType const & a , ViewType const & b ) {
      if constexpr (std::is_arithmetic_v<ViewType>) {
        return b >= 0 ? std::abs(a) : -std::abs(a);
      } else if constexpr (is_SArray<ViewType>) {
        ViewType ret;
        for (int i=0; i < a.size(); i++) {
          ret.data()[i] = b.data()[i] >= 0 ? std::abs(a.data()[i]) : -std::abs(a.data()[i]);
        }
        return ret;
      } else {
        auto ret = a.clone_object();
        Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                              Kokkos::RangePolicy<typename ViewType::execution_space>(0,a.size()) ,
                              KOKKOS_LAMBDA (int i) {
          ret.data()[i] = b.data()[i] >= 0 ? std::abs(a.data()[i]) : -std::abs(a.data()[i]);
        });
        return ret;
      }
    }


    // MERGE
    template <class V1, class V2>
    KOKKOS_INLINE_FUNCTION auto merge(V1 const & t, V1 const & f, V2 const & cond) {
      if constexpr (std::is_arithmetic_v<V1> && std::is_arithmetic_v<V2>) {
        return cond ? t : f;
      } else if constexpr (is_SArray<V1> && is_SArray<V2>) {
        V1 ret;
        for (int i=0; i < cond.size(); i++) {
          ret.data()[i] = cond.data()[i] ? t.data()[i] : f.data()[i];
        }
        return ret;
      } else {
        auto ret = t.clone_object();
        Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                              Kokkos::RangePolicy<typename V1::execution_space>(0,cond.size()) ,
                              KOKKOS_LAMBDA (int i) {
          ret.data()[i] = cond.data()[i] ? t.data()[i] : f.data()[i];
        });
        return ret;
      }
    }



    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // INTRINSICS
    //////////////////////////////////////////////////////////////////////////////////////////////////////

    template <class ViewType> KOKKOS_INLINE_FUNCTION auto           allocated (ViewType const & in) {
      return in.is_allocated();
    }
    template <class ViewType> KOKKOS_INLINE_FUNCTION auto           associated(ViewType const & in) {
      return in.is_allocated();
    }
    template <class ViewType> KOKKOS_INLINE_FUNCTION auto constexpr epsilon   (ViewType const & in) {
      if constexpr (std::is_floating_point_v<ViewType>) { 
        return std::numeric_limits<ViewType>::epsilon();
      } else {
        static_assert(std::is_floating_point_v<typename ViewType::value_type>,"ERROR: epsilon on non-floating-point type");
        return std::numeric_limits<typename ViewType::value_type>::epsilon();
      }
    }
    template <class ViewType> KOKKOS_INLINE_FUNCTION auto constexpr huge      (ViewType const & in) {
      if constexpr (std::is_arithmetic_v<ViewType>) {
        return std::numeric_limits<ViewType>::max();
      } else {
        return std::numeric_limits<typename ViewType::value_type>::max();
      }
    }
    template <class ViewType> KOKKOS_INLINE_FUNCTION auto constexpr tiny      (ViewType const & in) {
      if constexpr (std::is_floating_point_v<ViewType>) {
        return std::numeric_limits<ViewType>::min();
      } else {
        static_assert(std::is_floating_point_v<typename ViewType::value_type>,"ERROR: tiny on non-floating-point type");
        return std::numeric_limits<typename ViewType::value_type>::min();
      }
    }
    template <class ViewType> KOKKOS_INLINE_FUNCTION auto           size      (ViewType const & in) {
      return in.size();
    }
    template <class ViewType> KOKKOS_INLINE_FUNCTION auto           size      (ViewType const & in, std::integral auto i) {
      return in.extents()(i);
    }
    template <class ViewType> KOKKOS_INLINE_FUNCTION auto           ubound    (ViewType const & in) {
      return in.ubounds();
    }
    template <class ViewType> KOKKOS_INLINE_FUNCTION auto           ubound    (ViewType const & in, std::integral auto i) {
      return in.ubounds()(i);
    }
    template <class ViewType> KOKKOS_INLINE_FUNCTION auto           lbound    (ViewType const & in) {
      return in.lbounds();
    }
    template <class ViewType> KOKKOS_INLINE_FUNCTION auto           lbound    (ViewType const & in, std::integral auto i) {
      return in.lbounds()(i);
    }
    template <class ViewType> KOKKOS_INLINE_FUNCTION auto           shape     (ViewType const & in) {
      return in.extents();
    }



    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // REDUCTIONS
    //////////////////////////////////////////////////////////////////////////////////////////////////////

    // ANY
    template <class ViewType> inline auto any(ViewType const & in) {
      if constexpr (kokkos_debug) if (!in.span_is_contiguous()) Kokkos::abort("ERROR: any on non-contiguous View");
      if constexpr (kokkos_debug) if (!in.is_allocated      ()) Kokkos::abort("ERROR: any on unallocated View");
      if constexpr (is_SArray<ViewType>) {
        bool any_true = false;
        for (int i=0; i < in.size(); i++) { any_true = any_true || in.data()[i]; }
        return any_true;
      } else {
        ScalarLiveOut<bool> any_true(false);
        Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                              Kokkos::RangePolicy<typename ViewType::execution_space>(0,in.size()) ,
                              KOKKOS_LAMBDA (int i) {
          if (in.data()[i]) any_true = true;
        });
        return any_true.hostRead();
      }
    }



    // ALL
    template <class ViewType> inline auto all(ViewType const & in) {
      if constexpr (kokkos_debug) if (!in.span_is_contiguous()) Kokkos::abort("ERROR: all on non-contiguous View");
      if constexpr (kokkos_debug) if (!in.is_allocated      ()) Kokkos::abort("ERROR: all on unallocated View");
      if constexpr (is_SArray<ViewType>) {
        bool all_true = true;
        for (int i=0; i < in.size(); i++) { all_true = all_true && in.data()[i]; }
        return all_true;
      } else {
        ScalarLiveOut<bool> all_true(true);
        Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                              Kokkos::RangePolicy<typename ViewType::execution_space>(0,in.size()) ,
                              KOKKOS_LAMBDA (int i) {
          if (!in.data()[i]) all_true = false;
        });
        return all_true.hostRead();
      }
    }



    // SUM
    template <class ViewType> inline auto sum(ViewType const & in) {
      if constexpr (kokkos_debug) if (!in.span_is_contiguous()) Kokkos::abort("ERROR: sum on non-contiguous View");
      if constexpr (kokkos_debug) if (!in.is_allocated      ()) Kokkos::abort("ERROR: sum on unallocated View");
      using scalar_t = typename ViewType::non_const_value_type;
      if constexpr (is_SArray<ViewType>) {
        scalar_t result = in.data()[0];
        for (int i=1; i < in.size(); i++) { result += in.data()[i]; }
        return result;
      } else {
        scalar_t result;
        Kokkos::parallel_reduce( YAKL_AUTO_LABEL() ,
                                 Kokkos::RangePolicy<typename ViewType::execution_space>(0,in.size()) ,
                                 KOKKOS_LAMBDA (int i , scalar_t & lsum ) {
          lsum += in.data()[i];
        } , Kokkos::Sum<scalar_t>(result) );
        return result;
      }
    }



    // COUNT
    template <class ViewType> inline auto count(ViewType const & in) {
      if constexpr (kokkos_debug) if (!in.span_is_contiguous()) Kokkos::abort("ERROR: count on non-contiguous View");
      if constexpr (kokkos_debug) if (!in.is_allocated      ()) Kokkos::abort("ERROR: count on unallocated View");
      if constexpr (is_SArray<ViewType>) {
        size_t result = 0;
        for (int i=0; i < in.size(); i++) { if (in.data()[i]) result++; }
        return result;
      } else {
        yakl::Array<size_t *,typename ViewType::memory_space> num1d("num1d",in.size());
        Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                              Kokkos::RangePolicy<typename ViewType::execution_space>(0,in.size()) ,
                              KOKKOS_LAMBDA (int i) {
          num1d(i) = in.data()[i] ? 1 : 0;
        });
        return sum(num1d);
      }
    }



    // PRODUCT
    template <class ViewType> inline auto product(ViewType const & in) {
      if constexpr (kokkos_debug) if (!in.span_is_contiguous()) Kokkos::abort("ERROR: product on non-contiguous View");
      if constexpr (kokkos_debug) if (!in.is_allocated      ()) Kokkos::abort("ERROR: product on unallocated View");
      using scalar_t = typename ViewType::non_const_value_type;
      if constexpr (is_SArray<ViewType>) {
        scalar_t result = in.data()[0];
        for (int i=1; i < in.size(); i++) { result *= in.data()[i]; }
        return result;
      } else {
        scalar_t result;
        Kokkos::parallel_reduce( YAKL_AUTO_LABEL() ,
                                 Kokkos::RangePolicy<typename ViewType::execution_space>(0,in.size()) ,
                                 KOKKOS_LAMBDA (int i , scalar_t & lprod ) {
          lprod *= in.data()[i];
        } , Kokkos::Prod<scalar_t>(result) );
        return result;
      }
    }



    // MINVAL
    template <class ViewType> inline auto minval(ViewType const & in) {
      if constexpr (kokkos_debug) if (!in.span_is_contiguous()) Kokkos::abort("ERROR: minval on non-contiguous View");
      if constexpr (kokkos_debug) if (!in.is_allocated      ()) Kokkos::abort("ERROR: minval on unallocated View");
      using scalar_t = typename ViewType::non_const_value_type;
      if constexpr (is_SArray<ViewType>) {
        scalar_t result = in.data()[0];
        for (int i=1; i < in.size(); i++) { result = std::min(result,in.data()[i]); }
        return result;
      } else {
        scalar_t result;
        Kokkos::parallel_reduce( YAKL_AUTO_LABEL() ,
                                 Kokkos::RangePolicy<typename ViewType::execution_space>(0,in.size()) ,
                                 KOKKOS_LAMBDA (int i , scalar_t & lmin ) {
          lmin = std::min(lmin,in.data()[i]);
        } , Kokkos::Min<scalar_t>(result) );
        return result;
      }
    }



    // MAXVAL
    template <class ViewType> inline auto maxval(ViewType const & in) {
      if constexpr (kokkos_debug) if (!in.span_is_contiguous()) Kokkos::abort("ERROR: maxval on non-contiguous View");
      if constexpr (kokkos_debug) if (!in.is_allocated      ()) Kokkos::abort("ERROR: maxval on unallocated View");
      using scalar_t = typename ViewType::non_const_value_type;
      if constexpr (is_SArray<ViewType>) {
        scalar_t result = in.data()[0];
        for (int i=1; i < in.size(); i++) { result = std::max(result,in.data()[i]); }
        return result;
      } else {
        scalar_t result;
        Kokkos::parallel_reduce( YAKL_AUTO_LABEL() ,
                                 Kokkos::RangePolicy<typename ViewType::execution_space>(0,in.size()) ,
                                 KOKKOS_LAMBDA (int i , scalar_t & lmax ) {
          lmax = std::max(lmax,in.data()[i]);
        } , Kokkos::Max<scalar_t>(result) );
        return result;
      }
    }



    // MINLOC
    template <class ViewType> inline auto minloc(ViewType const & in) {
      if constexpr (kokkos_debug) if (!in.span_is_contiguous()) Kokkos::abort("ERROR: minloc on non-contiguous View");
      if constexpr (kokkos_debug) if (!in.is_allocated      ()) Kokkos::abort("ERROR: minloc on unallocated View");
      using scalar_t = typename ViewType::non_const_value_type;
      auto mn = minval(in);
      size_t iglob = 0;
      if constexpr (is_SArray<ViewType>) {
        for (int i=0; i < in.size(); i++) { if (in.data()[i] == mn) iglob = i; }
      } else {
        if constexpr (std::is_same_v<typename ViewType::memory_space,Kokkos::HostSpace>) {
          for (int i=0; i < in.size(); i++) { if (in.data()[i] == mn) iglob = i; }
        } else {
          ScalarLiveOut<size_t> iglob_slo(0);
          Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                                Kokkos::RangePolicy<typename ViewType::execution_space>(0,in.size()) ,
                                KOKKOS_LAMBDA (int i) {
            if (in.data()[i] == mn) iglob_slo = i;
          });
          iglob = iglob_slo.hostRead();
        }
      }
      return in.unpack_global_index(iglob);
    }



    // MAXLOC
    template <class ViewType> inline auto maxloc(ViewType const & in) {
      if constexpr (kokkos_debug) if (!in.span_is_contiguous()) Kokkos::abort("ERROR: maxloc on non-contiguous View");
      if constexpr (kokkos_debug) if (!in.is_allocated      ()) Kokkos::abort("ERROR: maxloc on unallocated View");
      using scalar_t = typename ViewType::non_const_value_type;
      auto mx = maxval(in);
      size_t iglob = 0;
      if constexpr (is_SArray<ViewType>) {
        for (int i=0; i < in.size(); i++) { if (in.data()[i] == mx) iglob = i; }
      } else {
        if constexpr (std::is_same_v<typename ViewType::memory_space,Kokkos::HostSpace>) {
          for (int i=0; i < in.size(); i++) { if (in.data()[i] == mx) iglob = i; }
        } else {
          ScalarLiveOut<size_t> iglob_slo(0);
          Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                                Kokkos::RangePolicy<typename ViewType::execution_space>(0,in.size()) ,
                                KOKKOS_LAMBDA (int i) {
            if (in.data()[i] == mx) iglob_slo = i;
          });
          iglob =  iglob_slo.hostRead();
        }
      }
      return in.unpack_global_index(iglob);
    }


  }
}


