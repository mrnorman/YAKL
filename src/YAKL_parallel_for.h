
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
    int  inner_size;
    YAKL_INLINE LaunchConfig() {
      b4b = false;
      inner_size = VL;
    }
    YAKL_INLINE LaunchConfig            (LaunchConfig const &rhs) {
      this->b4b=rhs.b4b;
      this->inner_size=rhs.inner_size;
    }
    YAKL_INLINE LaunchConfig            (LaunchConfig      &&rhs) {
      this->b4b=rhs.b4b;
      this->inner_size=rhs.inner_size;
    }
    YAKL_INLINE LaunchConfig & operator=(LaunchConfig const &rhs) {
      if (this == &rhs) return *this;
      this->b4b=rhs.b4b;
      this->inner_size=rhs.inner_size;
      return *this;
    }
    YAKL_INLINE LaunchConfig & operator=(LaunchConfig      &&rhs) {
      if (this == &rhs) return *this;
      this->b4b=rhs.b4b;
      this->inner_size=rhs.inner_size;
      return *this;
    }
    LaunchConfig enable_b4b() {
      #ifdef YAKL_B4B
        this->b4b = true;
      #endif
      return *this;
    }
    LaunchConfig set_inner_size(int num) {
      this->inner_size = num;
      return *this;
    }
  };


  using DefaultLaunchConfig = LaunchConfig<>;


  #include "YAKL_parallel_for_c.h"

  #include "YAKL_parallel_for_fortran.h"
}


