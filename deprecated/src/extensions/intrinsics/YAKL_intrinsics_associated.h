
#pragma once
// Included by YAKL_intrinsics.h

__YAKL_NAMESPACE_WRAPPER_BEGIN__
namespace yakl {
  namespace intrinsics {

    template <class T> YAKL_INLINE bool associated (T const &arr) { return arr.data() != nullptr; }

  }
}
__YAKL_NAMESPACE_WRAPPER_END__

