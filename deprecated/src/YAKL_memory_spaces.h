/**
 * @file
 * YAKL memory address space specifiers
 */

#pragma once

__YAKL_NAMESPACE_WRAPPER_BEGIN__
namespace yakl {

  // Labels for memory spaces. Only memDevice and memHost are expected to be used explicitly by the user
  /** @brief Specifies a device memory address space for a yakl::Array object */
  int constexpr memDevice = 1;
  /** @brief Specifies a device memory address space for a yakl::Array object */
  int constexpr memHost   = 2;
  #if defined(YAKL_ARCH_CUDA) || defined(YAKL_ARCH_HIP) || defined(YAKL_ARCH_SYCL)
    int constexpr memDefault = memDevice;
  #else
    /** @brief If the user does not specify a memory space template parameter to yakl::Array, host is the default. */
    int constexpr memDefault = memHost;
  #endif

}
__YAKL_NAMESPACE_WRAPPER_END__


