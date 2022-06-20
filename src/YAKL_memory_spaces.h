
#pragma once

namespace yakl {

  // Labels for memory spaces. Only memDevice and memHost are expected to be used explicitly by the user
  int constexpr memDevice = 1;
  int constexpr memHost   = 2;
  int constexpr memStack  = 3;
  #if defined(YAKL_ARCH_CUDA) || defined(YAKL_ARCH_HIP) || defined(YAKL_ARCH_SYCL)
    int constexpr memDefault = memDevice;
  #else
    int constexpr memDefault = memHost;
  #endif

}


