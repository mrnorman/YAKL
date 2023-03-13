
#pragma once
// Included by YAKL_intrinsics.h

__YAKL_NAMESPACE_WRAPPER_BEGIN__
namespace yakl {
  namespace intrinsics {

    template <class T> YAKL_INLINE auto shape(T const &arr) {
      #ifdef YAKL_DEBUG
        if (!allocated(arr)) yakl_throw("ERROR: Calling shape on an unallocated array");
      #endif
      return arr.get_dimensions();
    }

  }
}
__YAKL_NAMESPACE_WRAPPER_END__

