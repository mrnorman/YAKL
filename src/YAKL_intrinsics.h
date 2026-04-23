
#pragma once
// Included by YAKL.h

namespace yakl {
  namespace intrinsics {

    // ABS
    template <class ViewType> inline ViewType abs(ViewType const & in) {
      if constexpr (kokkos_debug) if (!in.span_is_contiguous()) Kokkos::abort("ERROR: abs on non-contiguous View");
      if constexpr (kokkos_debug) if (!in.is_allocated      ()) Kokkos::abort("ERROR: abs on unallocated View");
      if constexpr (is_SArray<ViewType>) {
        ViewType ret;
        for (int i=0; i < in.size(); i++) { ret.data()[i] = std::abs(in.data()[i]); }
        return ret;
      } else {
        auto ret = in.clone_object();
        if constexpr (yakl_auto_profile) timer_start("yakl::intrinsics::abs");
        Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                              Kokkos::RangePolicy<typename ViewType::execution_space>(0,in.size()) ,
                              KOKKOS_LAMBDA (size_t i) {
          ret.data()[i] = std::abs(in.data()[i]);
        } );
        if constexpr (yakl_auto_profile) timer_stop("yakl::intrinsics::abs");
        if constexpr (yakl_auto_fence) Kokkos::fence();
        return ret;
      }
    }



    // SIGN
    template <class ViewType>
    inline ViewType sign( ViewType const & a , ViewType const & b ) {
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
        if constexpr (yakl_auto_profile) timer_start("yakl::intrinsics::sign");
        Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                              Kokkos::RangePolicy<typename ViewType::execution_space>(0,a.size()) ,
                              KOKKOS_LAMBDA (size_t i) {
          ret.data()[i] = b.data()[i] >= 0 ? std::abs(a.data()[i]) : -std::abs(a.data()[i]);
        });
        if constexpr (yakl_auto_profile) timer_stop("yakl::intrinsics::sign");
        if constexpr (yakl_auto_fence) Kokkos::fence();
        return ret;
      }
    }



    // MERGE
    template <class V1, class V2>
    inline V1 merge(V1 const & t, V1 const & f, V2 const & cond) {
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
        if constexpr (yakl_auto_profile) timer_start("yakl::intrinsics::merge");
        Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                              Kokkos::RangePolicy<typename V1::execution_space>(0,cond.size()) ,
                              KOKKOS_LAMBDA (size_t i) {
          ret.data()[i] = cond.data()[i] ? t.data()[i] : f.data()[i];
        });
        if constexpr (yakl_auto_profile) timer_stop("yakl::intrinsics::merge");
        if constexpr (yakl_auto_fence) Kokkos::fence();
        return ret;
      }
    }



    // MATMUL_RC
    template <class V1, class V2 >requires is_SArray<V1> && is_SArray<V2> && V1::is_cstyle && V2::is_cstyle
    KOKKOS_INLINE_FUNCTION auto matmul_rc(V1 const & v1 , V2 const & v2) {
      static_assert((V1::rank==2) && (V2::rank <= 2),"ERROR: matmul_rc on incompatible rank SArray types");
      int constexpr nrows1 = V1::template extent<0>();
      int constexpr ncols1 = V1::template extent<1>();
      int constexpr nrows2 = V2::template extent<0>();
      static_assert(ncols1 == nrows2,"ERROR: matmul_rc matrix-vector with incompatible dimension sizes");
      using T = std::remove_cv_t<decltype(v1.data()[0]+v2.data()[0])>;
      if constexpr (V2::rank==1) {
        SArray<T,nrows1> ret;
        for (int ir1 = 0; ir1 < nrows1; ir1++) {
          ret(ir1) = static_cast<T>(0);
          for (int ic1 = 0; ic1 < ncols1; ic1++) { ret(ir1) += v1(ir1,ic1)*v2(ic1); }
        }
        return ret;
      } else {
        int constexpr ncols2 = V2::template extent<1>();
        SArray<T,nrows1,ncols2> ret;
        for (int ir1 = 0; ir1 < nrows1; ir1++) {
          for (int ic2 = 0; ic2 < ncols2; ic2++) {
            ret(ir1,ic2) = static_cast<T>(0);
            for (int ic1 = 0; ic1 < ncols1; ic1++) { ret(ir1,ic2) += v1(ir1,ic1)*v2(ic1,ic2); }
          }
        }
        return ret;
      }
    }



    // MATMUL_CR
    template <class V1, class V2 >requires is_SArray<V1> && is_SArray<V2> && V1::is_cstyle && V2::is_cstyle
    KOKKOS_INLINE_FUNCTION auto matmul_cr(V1 const & v1 , V2 const & v2) {
      static_assert((V1::rank==2) && (V2::rank <= 2),"ERROR: matmul_cr on incompatible rank SArray types");
      int constexpr nrows1 = V1::template extent<1>();
      int constexpr ncols1 = V1::template extent<0>();
      using T = std::remove_cv_t<decltype(v1.data()[0]+v2.data()[0])>;
      if constexpr (V2::rank==1) {
        int constexpr nrows2 = V2::template extent<0>();
        static_assert(ncols1 == nrows2,"ERROR: matmul_cr matrix-vector with incompatible dimension sizes");
        SArray<T,nrows1> ret;
        for (int ir1 = 0; ir1 < nrows1; ir1++) {
          ret(ir1) = static_cast<T>(0);
          for (int ic1 = 0; ic1 < ncols1; ic1++) { ret(ir1) += v1(ic1,ir1)*v2(ic1); }
        }
        return ret;
      } else {
        int constexpr nrows2 = V2::template extent<1>();
        static_assert(ncols1 == nrows2,"ERROR: matmul_cr matrix-vector with incompatible dimension sizes");
        int constexpr ncols2 = V2::template extent<0>();
        SArray<T,ncols2,nrows1> ret;
        for (int ir1 = 0; ir1 < nrows1; ir1++) {
          for (int ic2 = 0; ic2 < ncols2; ic2++) {
            ret(ic2,ir1) = static_cast<T>(0);
            for (int ic1 = 0; ic1 < ncols1; ic1++) { ret(ic2,ir1) += v1(ic1,ir1)*v2(ic2,ic1); }
          }
        }
        return ret;
      }
    }



    // TRANSPOSE
    template <class V> requires is_SArray<V> && (V::rank == 2) && V::is_cstyle
    KOKKOS_INLINE_FUNCTION auto transpose(V const & v) {
      int constexpr nr = V::template extent<0>();
      int constexpr nc = V::template extent<1>();
      SArray<typename V::non_const_value_type,nc,nr> ret;
      for (int ir = 0; ir < nr; ir++) {
        for (int ic = 0; ic < nc; ic++) { ret(ic,ir) = v(ir,ic); }
      }
      return ret;
    }



    template <class V> requires is_SArray<V> && (V::rank == 2) && V::is_cstyle
    KOKKOS_INLINE_FUNCTION auto matinv(V const & a) {
      static_assert(V::template extent<0>() == V::template extent<1>(),"ERROR: matinv on non-square matrix");
      int constexpr n = V::template extent<0>();
      using T = typename V::non_const_value_type;
      SArray<T,n,n> scratch;
      SArray<T,n,n> inv;
      // Initialize scratch as copy of a, inv as identity
      for (int irow = 0; irow < n; irow++) {
        for (int icol = 0; icol < n; icol++) {
          scratch(irow,icol) = a(irow,icol);
          inv    (irow,icol) = (irow==icol) ? static_cast<T>(1) : static_cast<T>(0);
        }
      }
      // Forward elimination with partial pivoting
      for (int idiag = 0; idiag < n; idiag++) {
        // Find pivot row: largest absolute value in column idiag from idiag downward
        int pivot     = idiag;
        T   pivot_val = std::abs(scratch(idiag,idiag));
        for (int irow = idiag+1; irow < n; irow++) {
          T val = std::abs(scratch(irow,idiag));
          if (val > pivot_val) { pivot_val = val; pivot = irow; }
        }
        // Swap pivot row into diagonal position
        if (pivot != idiag) {
          for (int icol = 0; icol < n; icol++) {
            T tmp = scratch(idiag,icol); scratch(idiag,icol) = scratch(pivot,icol); scratch(pivot,icol) = tmp;
            tmp   = inv    (idiag,icol); inv    (idiag,icol) = inv    (pivot,icol); inv    (pivot,icol) = tmp;
          }
        }
        // Normalize pivot row
        T factor = static_cast<T>(1) / scratch(idiag,idiag);
        for (int icol = idiag; icol < n; icol++) { scratch(idiag,icol) *= factor; }
        for (int icol = 0;     icol < n; icol++) { inv    (idiag,icol) *= factor; }
        // Eliminate below
        for (int irow = idiag+1; irow < n; irow++) {
          T fac = scratch(irow,idiag);
          for (int icol = idiag; icol < n; icol++) { scratch(irow,icol) -= fac * scratch(idiag,icol); }
          for (int icol = 0;     icol < n; icol++) { inv    (irow,icol) -= fac * inv    (idiag,icol); }
        }
      }
      // Back substitution to zero out upper triangle
      for (int idiag = n-1; idiag >= 1; idiag--) {
        for (int irow = 0; irow < idiag; irow++) {
          T fac = scratch(irow,idiag);
          for (int icol = irow+1; icol < n; icol++) { scratch(irow,icol) -= fac * scratch(idiag,icol); }
          for (int icol = 0;      icol < n; icol++) { inv    (irow,icol) -= fac * inv    (idiag,icol); }
        }
      }
      return inv;
    }




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



    // ANY
    template <class ViewType> inline bool any(ViewType const & in) {
      if constexpr (kokkos_debug) if (!in.span_is_contiguous()) Kokkos::abort("ERROR: any on non-contiguous View");
      if constexpr (kokkos_debug) if (!in.is_allocated      ()) Kokkos::abort("ERROR: any on unallocated View");
      if constexpr (is_SArray<ViewType>) {
        bool any_true = false;
        for (int i=0; i < in.size(); i++) { any_true = any_true || in.data()[i]; }
        return any_true;
      } else {
        ScalarLiveOut<bool> any_true(false);
        if constexpr (yakl_auto_profile) timer_start("yakl::intrinsics::any");
        Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                              Kokkos::RangePolicy<typename ViewType::execution_space>(0,in.size()) ,
                              KOKKOS_LAMBDA (size_t i) {
          if (in.data()[i]) any_true = true;
        });
        if constexpr (yakl_auto_profile) timer_stop("yakl::intrinsics::any");
        if constexpr (yakl_auto_fence) Kokkos::fence();
        return any_true.hostRead();
      }
    }



    // ALL
    template <class ViewType> inline bool all(ViewType const & in) {
      if constexpr (kokkos_debug) if (!in.span_is_contiguous()) Kokkos::abort("ERROR: all on non-contiguous View");
      if constexpr (kokkos_debug) if (!in.is_allocated      ()) Kokkos::abort("ERROR: all on unallocated View");
      if constexpr (is_SArray<ViewType>) {
        bool all_true = true;
        for (int i=0; i < in.size(); i++) { all_true = all_true && in.data()[i]; }
        return all_true;
      } else {
        ScalarLiveOut<bool> all_true(true);
        if constexpr (yakl_auto_profile) timer_start("yakl::intrinsics::all");
        Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                              Kokkos::RangePolicy<typename ViewType::execution_space>(0,in.size()) ,
                              KOKKOS_LAMBDA (size_t i) {
          if (!in.data()[i]) all_true = false;
        });
        if constexpr (yakl_auto_profile) timer_stop("yakl::intrinsics::all");
        if constexpr (yakl_auto_fence) Kokkos::fence();
        return all_true.hostRead();
      }
    }



    // SUM
    template <class ViewType> inline typename ViewType::non_const_value_type sum(ViewType const & in) {
      if constexpr (kokkos_debug) if (!in.span_is_contiguous()) Kokkos::abort("ERROR: sum on non-contiguous View");
      if constexpr (kokkos_debug) if (!in.is_allocated      ()) Kokkos::abort("ERROR: sum on unallocated View");
      using scalar_t = typename ViewType::non_const_value_type;
      if constexpr (is_SArray<ViewType>) {
        scalar_t result = in.data()[0];
        for (int i=1; i < in.size(); i++) { result += in.data()[i]; }
        return result;
      } else {
        scalar_t result;
        if constexpr (yakl_auto_profile) timer_start("yakl::intrinsics::sum");
        Kokkos::parallel_reduce( YAKL_AUTO_LABEL() ,
                                 Kokkos::RangePolicy<typename ViewType::execution_space>(0,in.size()) ,
                                 KOKKOS_LAMBDA (size_t i , scalar_t & lsum ) {
          lsum += in.data()[i];
        } , Kokkos::Sum<scalar_t>(result) );
        if constexpr (yakl_auto_profile) timer_stop("yakl::intrinsics::sum");
        if constexpr (yakl_auto_fence) Kokkos::fence();
        return result;
      }
    }



    // COUNT
    template <class ViewType> inline size_t count(ViewType const & in) {
      if constexpr (kokkos_debug) if (!in.span_is_contiguous()) Kokkos::abort("ERROR: count on non-contiguous View");
      if constexpr (kokkos_debug) if (!in.is_allocated      ()) Kokkos::abort("ERROR: count on unallocated View");
      if constexpr (is_SArray<ViewType>) {
        size_t result = 0;
        for (int i=0; i < in.size(); i++) { if (in.data()[i]) result++; }
        return result;
      } else {
        yakl::Array<size_t *,typename ViewType::memory_space> num1d("num1d",in.size());
        if constexpr (yakl_auto_profile) timer_start("yakl::intrinsics::count");
        Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                              Kokkos::RangePolicy<typename ViewType::execution_space>(0,in.size()) ,
                              KOKKOS_LAMBDA (size_t i) {
          num1d(i) = in.data()[i] ? 1 : 0;
        });
        if constexpr (yakl_auto_profile) timer_stop("yakl::intrinsics::count");
        if constexpr (yakl_auto_fence) Kokkos::fence();
        return sum(num1d);
      }
    }



    // PRODUCT
    template <class ViewType> inline typename ViewType::non_const_value_type product(ViewType const & in) {
      if constexpr (kokkos_debug) if (!in.span_is_contiguous()) Kokkos::abort("ERROR: product on non-contiguous View");
      if constexpr (kokkos_debug) if (!in.is_allocated      ()) Kokkos::abort("ERROR: product on unallocated View");
      using scalar_t = typename ViewType::non_const_value_type;
      if constexpr (is_SArray<ViewType>) {
        scalar_t result = in.data()[0];
        for (int i=1; i < in.size(); i++) { result *= in.data()[i]; }
        return result;
      } else {
        scalar_t result;
        if constexpr (yakl_auto_profile) timer_start("yakl::intrinsics::product");
        Kokkos::parallel_reduce( YAKL_AUTO_LABEL() ,
                                 Kokkos::RangePolicy<typename ViewType::execution_space>(0,in.size()) ,
                                 KOKKOS_LAMBDA (size_t i , scalar_t & lprod ) {
          lprod *= in.data()[i];
        } , Kokkos::Prod<scalar_t>(result) );
        if constexpr (yakl_auto_profile) timer_stop("yakl::intrinsics::product");
        if constexpr (yakl_auto_fence) Kokkos::fence();
        return result;
      }
    }



    // MINVAL
    template <class ViewType> inline typename ViewType::non_const_value_type minval(ViewType const & in) {
      if constexpr (kokkos_debug) if (!in.span_is_contiguous()) Kokkos::abort("ERROR: minval on non-contiguous View");
      if constexpr (kokkos_debug) if (!in.is_allocated      ()) Kokkos::abort("ERROR: minval on unallocated View");
      using scalar_t = typename ViewType::non_const_value_type;
      if constexpr (is_SArray<ViewType>) {
        scalar_t result = in.data()[0];
        for (int i=1; i < in.size(); i++) { result = std::min(result,in.data()[i]); }
        return result;
      } else {
        scalar_t result;
        if constexpr (yakl_auto_profile) timer_start("yakl::intrinsics::minval");
        Kokkos::parallel_reduce( YAKL_AUTO_LABEL() ,
                                 Kokkos::RangePolicy<typename ViewType::execution_space>(0,in.size()) ,
                                 KOKKOS_LAMBDA (size_t i , scalar_t & lmin ) {
          lmin = std::min(lmin,in.data()[i]);
        } , Kokkos::Min<scalar_t>(result) );
        if constexpr (yakl_auto_profile) timer_stop("yakl::intrinsics::minval");
        if constexpr (yakl_auto_fence) Kokkos::fence();
        return result;
      }
    }



    // MAXVAL
    template <class ViewType> inline typename ViewType::non_const_value_type maxval(ViewType const & in) {
      if constexpr (kokkos_debug) if (!in.span_is_contiguous()) Kokkos::abort("ERROR: maxval on non-contiguous View");
      if constexpr (kokkos_debug) if (!in.is_allocated      ()) Kokkos::abort("ERROR: maxval on unallocated View");
      using scalar_t = typename ViewType::non_const_value_type;
      if constexpr (is_SArray<ViewType>) {
        scalar_t result = in.data()[0];
        for (int i=1; i < in.size(); i++) { result = std::max(result,in.data()[i]); }
        return result;
      } else {
        scalar_t result;
        if constexpr (yakl_auto_profile) timer_start("yakl::intrinsics::maxval");
        Kokkos::parallel_reduce( YAKL_AUTO_LABEL() ,
                                 Kokkos::RangePolicy<typename ViewType::execution_space>(0,in.size()) ,
                                 KOKKOS_LAMBDA (size_t i , scalar_t & lmax ) {
          lmax = std::max(lmax,in.data()[i]);
        } , Kokkos::Max<scalar_t>(result) );
        if constexpr (yakl_auto_profile) timer_stop("yakl::intrinsics::maxval");
        if constexpr (yakl_auto_fence) Kokkos::fence();
        return result;
      }
    }



    // MINLOC
    template <class ViewType> inline auto minloc(ViewType const & in) ->
    decltype(in.unpack_global_index(0))
    {
      if constexpr (kokkos_debug) if (!in.span_is_contiguous()) Kokkos::abort("ERROR: minloc on non-contiguous View");
      if constexpr (kokkos_debug) if (!in.is_allocated      ()) Kokkos::abort("ERROR: minloc on unallocated View");
      using scalar_t = typename ViewType::non_const_value_type;
      auto mn = minval(in);
      size_t iglob = 0;
      if constexpr (is_SArray<ViewType>) {
        for (int i=0; i < in.size(); i++) { if (in.data()[i] == mn) iglob = i; }
      } else {
        if constexpr (std::is_same_v<typename ViewType::memory_space,Kokkos::HostSpace>) {
          for (size_t i=0; i < in.size(); i++) { if (in.data()[i] == mn) iglob = i; }
        } else {
          ScalarLiveOut<size_t> iglob_slo(0);
          if constexpr (yakl_auto_profile) timer_start("yakl::intrinsics::minloc");
          Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                                Kokkos::RangePolicy<typename ViewType::execution_space>(0,in.size()) ,
                                KOKKOS_LAMBDA (size_t i) {
            if (in.data()[i] == mn) iglob_slo = i;
          });
          if constexpr (yakl_auto_profile) timer_stop("yakl::intrinsics::minloc");
          if constexpr (yakl_auto_fence) Kokkos::fence();
          iglob = iglob_slo.hostRead();
        }
      }
      return in.unpack_global_index(iglob);
    }



    // MAXLOC
    template <class ViewType> inline auto maxloc(ViewType const & in) ->
    decltype(in.unpack_global_index(0))
    {
      if constexpr (kokkos_debug) if (!in.span_is_contiguous()) Kokkos::abort("ERROR: maxloc on non-contiguous View");
      if constexpr (kokkos_debug) if (!in.is_allocated      ()) Kokkos::abort("ERROR: maxloc on unallocated View");
      using scalar_t = typename ViewType::non_const_value_type;
      auto mx = maxval(in);
      size_t iglob = 0;
      if constexpr (is_SArray<ViewType>) {
        for (int i=0; i < in.size(); i++) { if (in.data()[i] == mx) iglob = i; }
      } else {
        if constexpr (std::is_same_v<typename ViewType::memory_space,Kokkos::HostSpace>) {
          for (size_t i=0; i < in.size(); i++) { if (in.data()[i] == mx) iglob = i; }
        } else {
          ScalarLiveOut<size_t> iglob_slo(0);
          if constexpr (yakl_auto_profile) timer_start("yakl::intrinsics::maxloc");
          Kokkos::parallel_for( YAKL_AUTO_LABEL() ,
                                Kokkos::RangePolicy<typename ViewType::execution_space>(0,in.size()) ,
                                KOKKOS_LAMBDA (size_t i) {
            if (in.data()[i] == mx) iglob_slo = i;
          });
          if constexpr (yakl_auto_profile) timer_stop("yakl::intrinsics::maxloc");
          if constexpr (yakl_auto_fence) Kokkos::fence();
          iglob =  iglob_slo.hostRead();
        }
      }
      return in.unpack_global_index(iglob);
    }


  }
}


