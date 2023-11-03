/**
 * @file
 * YAKL kernel launch configuration class
 */

#pragma once


__YAKL_NAMESPACE_WRAPPER_BEGIN__
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
      /** @brief Each hardware backend has its own default length for inner loop sizes */
      #define YAKL_DEFAULT_VECTOR_LEN 128
    #endif
  #endif


  // Empty launch config struct to set the vector length for a kernel launch
  /**
   * @brief This class informs YAKL `parallel_for` and `parallel_outer` routines how to launch kernels.
   * 
   * It contains two optional template parameters: (1) `VL`: When passed to `parallel_for`, this
   * defines the inner looping size on the device (e.g. "block size" for CUDA and HIP. When passed to
   * `parallel_outer`, this defines the maximum inner looping size on the device. (2) `B4B`: If this
   * is set to `true`, then this tells `parallel_for` and `parallel_outer` to run the kernel serially
   * (only when the `-DYAKL_B4B` CPP macro is defined) to enable bitwise determinism when desired 
   * for kernels with yakl::atomicAdd in them.
   * 
   * @param VL  For `parallel_for`, this is the inner looping size. For `parallel_outer`, this is the
   *            **maximum** inner looping size.
   * @param B4B If the CPP macro `YAKL_B4B` is also defined, `B4B == true` will force the kernel to run
   *            in serial, typically used for kernels that contain yakl::atomicAdd to maintain bitwise
   *            determinism run-to-run. If `YAKL_B4B` is not defined, the kernel runs normally.
   */
  template <int VL = YAKL_DEFAULT_VECTOR_LEN, bool B4B = false>
  struct LaunchConfig {
  public:
    /** @private */
    int  inner_size;
    /** @private */
    Stream stream;
    /** @private */
    size_t inner_cache_bytes;
    /** @brief set_inner_size() defaults to YAKL_DEFAULT_VECTOR_LEN */
    LaunchConfig () { inner_size = VL; inner_cache_bytes = 0; }
    ~LaunchConfig() { inner_size = VL; inner_cache_bytes = 0; }
    /** @brief LaunchConfig objects may be copied or moved. */
    LaunchConfig            (LaunchConfig const &rhs) { copyfrom(rhs); }
    /** @brief LaunchConfig objects may be copied or moved. */
    LaunchConfig            (LaunchConfig      &&rhs) { copyfrom(rhs); }
    /** @brief LaunchConfig objects may be copied or moved. */
    LaunchConfig & operator=(LaunchConfig const &rhs) { copyfrom(rhs); return *this; }
    /** @brief LaunchConfig objects may be copied or moved. */
    LaunchConfig & operator=(LaunchConfig      &&rhs) { copyfrom(rhs); return *this; }
    void copyfrom(LaunchConfig const &rhs) {
      this->inner_size = rhs.inner_size;
      this->stream = rhs.stream;
      this->inner_cache_bytes = rhs.inner_cache_bytes;
    }
    /** @brief This sets the **actual** inner looping size whereas the template parameter `VL` sets the maximum
     *         inner looping size. */
    LaunchConfig set_inner_size(int num) {
      if (num <= 0 ) { throw std::runtime_error("ERROR: set_inner_size(int) parameter was <= 0"); }
      if (num >  VL) { throw std::runtime_error("ERROR: set_inner_size(int) parameter larger than templated VecLen"); }
      this->inner_size = num;
      return *this;
    }
    /** @brief Get the inner loop size for hierarchical parallelism. */
    int get_inner_size() const { return this->inner_size; }
    /** @brief Set the stream in which this launch will run. */
    LaunchConfig set_stream(Stream stream) { this->stream = stream; return *this; }
    /** @brief Get the stream in which this launch will run. */
    Stream get_stream() const { return this->stream; }
    /** @brief Set the stream in which this launch will run. */
    LaunchConfig set_inner_cache_bytes(size_t bytes) { this->inner_cache_bytes = bytes; return *this; }
    /** @brief Get the stream in which this launch will run. */
    size_t get_inner_cache_bytes() const { return this->inner_cache_bytes; }
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
   * @brief This class is necessary for coordination of two-level parallelism.
   * 
   * A yakl::InnerHandler object must be
   * accepted as a parameter in the functor passed to `parallel_outer`, and it must be passed as a
   * parameter to `parallel_inner`, `fence_inner`, and `single_inner`. An object of this class should never
   * need to be explicitly created by the user.
   */
  #if defined(YAKL_ARCH_SYCL)
    // This wraps a pointer to a sycl::nd_item<1> object for the SYCL backend's outer,inner parallelism
    // The object is guaranteed to stay in scope throughout a call to parallel_outer, which is the duration
    // of this class's use. Therefore, this behavior is safe to assume.
    // This is necessary because SYCL unfortunately doesn't expose a constructor for nd_item<1>
    class InnerHandler {
    public:
      sycl::nd_item<1> const * ptr;
      size_t                   inner_cache_bytes;
      int                      inner_size;
      char                   * slm;
      YAKL_INLINE InnerHandler() { ptr = nullptr; inner_cache_bytes = 0; slm = nullptr; inner_size = 0; }
      YAKL_INLINE explicit InnerHandler(sycl::nd_item<1> const &item, size_t inner_cache_bytes, void *slm) {
        this->ptr               = &item;
        this->inner_cache_bytes = inner_cache_bytes;
        this->slm               = static_cast<char *>(slm);
        YAKL_EXECUTE_ON_DEVICE_ONLY( this->inner_size = item->get_local_range(0); )
      }
      YAKL_INLINE sycl::nd_item<1> const * get_item() const { return ptr; }
      YAKL_INLINE int    get_inner_size       () const { return inner_size; }
      YAKL_INLINE size_t get_inner_cache_bytes() const { return inner_cache_bytes; }
      YAKL_INLINE void   inner_barrier        () const {
        YAKL_EXECUTE_ON_DEVICE_ONLY( item->barrier(sycl::access::fence_space::local_space) );
      }
      template <class T>
      YAKL_INLINE T * get_inner_cache_pointer(size_t offset_bytes = 0) const {
        if (slm == nullptr) return nullptr;
        return static_cast<T *>( static_cast<void *>( &(slm[offset_bytes]) ) );
      }
    };
  #elif defined(YAKL_ARCH_CUDA) || defined(YAKL_ARCH_HIP)
    struct InnerHandler {
      int    inner_size;
      size_t inner_cache_bytes;
      YAKL_INLINE InnerHandler() { inner_cache_bytes = 0; inner_size = 0; }
      YAKL_INLINE InnerHandler(int inner_size, size_t inner_cache_bytes) {
        this->inner_size        = inner_size;
        this->inner_cache_bytes = inner_cache_bytes;
      }
      YAKL_INLINE int    get_inner_size       () const { return inner_size; }
      YAKL_INLINE size_t get_inner_cache_bytes() const { return inner_cache_bytes; }
      YAKL_INLINE void   inner_barrier        () const { YAKL_EXECUTE_ON_DEVICE_ONLY( __syncthreads() ); }
      template <class T>
      YAKL_INLINE T * get_inner_cache_pointer(size_t offset_bytes = 0) const {
        YAKL_EXECUTE_ON_DEVICE_ONLY(
          extern __shared__ char slm[];
          if (slm == nullptr) return nullptr;
          return static_cast<T *>( static_cast<void *>( &(slm[offset_bytes]) ) );
        )
        YAKL_EXECUTE_ON_HOST_ONLY( return nullptr; )
      }
    };
  #else
    struct InnerHandler {
      char   * slm;
      int      inner_size;
      size_t   inner_cache_bytes;
      YAKL_INLINE InnerHandler( int inner_size = 0 , size_t inner_cache_bytes = 0 , void * slm = nullptr ) {
        this->inner_size        = inner_size       ;
        this->inner_cache_bytes = inner_cache_bytes;
        this->slm               = static_cast<char *>(slm);
      }
      YAKL_INLINE int    get_inner_size       () const { return inner_size ; }
      YAKL_INLINE size_t get_inner_cache_bytes() const { return inner_cache_bytes; }
      YAKL_INLINE void   inner_barrier        () const { }
      template <class T>
      YAKL_INLINE T * get_inner_cache_pointer(size_t offset_bytes = 0) const {
        if (slm == nullptr) return nullptr;
        return static_cast<T *>( static_cast<void *>( &(slm[offset_bytes]) ) );
      }
    };
  #endif
}
__YAKL_NAMESPACE_WRAPPER_END__


