
#pragma once
// Included by YAKL_intrinsics.h

__YAKL_NAMESPACE_WRAPPER_BEGIN__
namespace yakl {
  namespace intrinsics {

    template <class T> YAKL_INLINE int  lbound (T const &arr, int dim) {
      #ifdef YAKL_DEBUG
        if ( ! allocated( arr ) ) yakl_throw("ERROR: calling lbound on an unallocated array");
      #endif
      return arr.get_lbounds()(dim);
    }

    template <class T> YAKL_INLINE auto lbound (T const &arr) { 
      #ifdef YAKL_DEBUG
        if ( ! allocated( arr ) ) yakl_throw("ERROR: calling lbound on an unallocated array");
      #endif
      return arr.get_lbounds();
    }

  }
}
__YAKL_NAMESPACE_WRAPPER_END__

