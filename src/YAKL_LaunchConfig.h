
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
  /**
   * @brief This class informs YAKL `parallel_for` and `parallel_outer` routines how to launch kernels.
   *        It contains two optional template parameters: (1) `VL`: When passed to `parallel_for`, this
   *        defines the inner looping size on the device (e.g. "block size" for CUDA and HIP. When passed to
   *        `parallel_outer`, this defines the maximum inner looping size on the device. (2) `B4B`: If this
   *        is set to `true`, then this tells `parallel_for` and `parallel_outer` to run the kernel serially
   *        (only when the `-DYAKL_B4B` CPP macro is defined) to enable bitwise determinism when desired 
   *        for kernels with yakl::atomicAdd in them.
   */
  template <int VL = YAKL_DEFAULT_VECTOR_LEN, bool B4B = false>
  struct LaunchConfig {
  public:
    int  inner_size;
    YAKL_INLINE LaunchConfig() { inner_size = VL; }
    YAKL_INLINE ~LaunchConfig() { inner_size = VL; }
    YAKL_INLINE LaunchConfig            (LaunchConfig const &rhs) { this->inner_size = rhs.inner_size; }
    YAKL_INLINE LaunchConfig            (LaunchConfig      &&rhs) { this->inner_size = rhs.inner_size; }
    YAKL_INLINE LaunchConfig & operator=(LaunchConfig const &rhs) { this->inner_size = rhs.inner_size; return *this; }
    YAKL_INLINE LaunchConfig & operator=(LaunchConfig      &&rhs) { this->inner_size = rhs.inner_size; return *this; }
    /**
     * @brief This sets the **actuall** inner looping size whereas the template parameter `VL` sets the maximum
     *        inner looping size.
     */
    LaunchConfig set_inner_size(int num) { this->inner_size = num; return *this; }
  };


  /**
   * @brief This launch configuration sets vector length to the device default and `B4B` to `false`.
   */
  using DefaultLaunchConfig = LaunchConfig<>;

  /**
   * @brief launch configuration sets B4B == true with a user-specified `VecLen`.
   */
  template <int VecLen=YAKL_DEFAULT_VECTOR_LEN> using LaunchConfigB4b = LaunchConfig<VecLen,true>;

  /**
   * @brief launch configuration sets B4B == true with the default `VecLen`.
   */
  using DefaultLaunchConfigB4b = LaunchConfig<YAKL_DEFAULT_VECTOR_LEN,true>;


  /**
   * @brief This class is necessary for coordination of two-level parallelism. A yakl::InnerHandler object must be
   *        accepted as a parameter in the functor passed to `parallel_outer`, and it must be passed as a
   *        parameter to `parallel_inner`, `fence_inner`, and `single_inner`. An object of this class should never
   *        need to be explicitly created by the user.
   */
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


