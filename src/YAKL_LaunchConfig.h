
#pragma once


namespace yakl {
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
    YAKL_INLINE ~LaunchConfig() {
      b4b = false;
      inner_size = VL;
    }
    YAKL_INLINE LaunchConfig            (LaunchConfig const &rhs) {
      this->b4b = rhs.b4b;
      this->inner_size = rhs.inner_size;
    }
    YAKL_INLINE LaunchConfig            (LaunchConfig      &&rhs) {
      this->b4b = rhs.b4b;
      this->inner_size = rhs.inner_size;
    }
    YAKL_INLINE LaunchConfig & operator=(LaunchConfig const &rhs) {
      if (this == &rhs) return *this;
      this->b4b = rhs.b4b;
      this->inner_size = rhs.inner_size;
      return *this;
    }
    YAKL_INLINE LaunchConfig & operator=(LaunchConfig      &&rhs) {
      if (this == &rhs) return *this;
      this->b4b = rhs.b4b;
      this->inner_size = rhs.inner_size;
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


  #if defined(YAKL_ARCH_SYCL)
    // This wraps a pointer to a sycl::nd_item<1> object for the SYCL backend's outer,inner parallelism
    // The object is guaranteed to stay in scope throughout a call to parallel_outer, which is the duration
    // of this class's use. Therefore, this behavior is safe to assume.
    // This is necessary because SYCL unfortunately doesn't expose a constructor for nd_item<1>
    class InnerHandler {
    public:
      sycl::nd_item<1> const *ptr;
      YAKL_INLINE InnerHandler() { ptr = nullptr; }
      YAKL_INLINE explicit InnerHandler(sycl::nd_item<1> const &item) { this->ptr = &item; }
      YAKL_INLINE sycl::nd_item<1> get_item() const { return *ptr; }
    };
  #else
    typedef struct InnerHandlerEmpty {} InnerHandler;
  #endif
}


