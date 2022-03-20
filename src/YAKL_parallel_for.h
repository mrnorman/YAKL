
#pragma once
// Included by YAKL.h


namespace yakl {
  // Useful for slightly faster modulo when computing indices from global thread IDs
  template <class T> constexpr T fastmod(T a , T b) {
    return a < b ? a : a-b*(a/b);
  }

  // Set default vector lengths. On GPU devices, this is the block size
  #ifndef YAKL_DEFAULT_VECTOR_LEN
    #if   defined(YAKL_ARCH_CUDA)
      #define YAKL_DEFAULT_VECTOR_LEN 128
    #elif defined(YAKL_ARCH_HIP)
      #define YAKL_DEFAULT_VECTOR_LEN 256
    #elif defined(YAKL_ARCH_SYCL)
      #define YAKL_DEFAULT_VECTOR_LEN 128
    #else
      #define YAKL_DEFAULT_VECTOR_LEN 128
    #endif
  #endif


  // Empty launch config struct to set the vector length for a kernel launch
  template <int VL = YAKL_DEFAULT_VECTOR_LEN>
  struct LaunchConfig {
  public:
    bool b4b;
    LaunchConfig() {b4b = false;}
    LaunchConfig            (LaunchConfig const &config) = default;
    LaunchConfig            (LaunchConfig      &&config) = default;
    LaunchConfig & operator=(LaunchConfig const &config) = default;
    LaunchConfig & operator=(LaunchConfig      &&config) = default;
    LaunchConfig enable_b4b() {
      #ifdef YAKL_B4B
        this->b4b = true;
      #endif
      return *this;
    }
  };


  using DefaultLaunchConfig = LaunchConfig<>;


  #include "YAKL_parallel_for_c.h"

  #include "YAKL_parallel_for_fortran.h"
}


