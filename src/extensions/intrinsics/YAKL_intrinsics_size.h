
#pragma once
// Included by YAKL_intrinsics.h

__YAKL_NAMESPACE_WRAPPER_BEGIN__
namespace yakl {
  namespace intrinsics {

    template <class T> YAKL_INLINE int  size(T const &arr, int dim) {
      #ifdef YAKL_DEBUG
        if (!allocated(arr)) yakl_throw("ERROR: Calling size on an unallocated array");
      #endif
      return arr.get_dimensions()(dim);
    }

    template <class T> YAKL_INLINE int  size(T const &arr) {
      #ifdef YAKL_DEBUG
        if (!allocated(arr)) yakl_throw("ERROR: Calling size on an unallocated array");
      #endif
      return arr.totElems();
    }

  }
}
__YAKL_NAMESPACE_WRAPPER_END__

