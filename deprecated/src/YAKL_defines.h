/**
 * @file
 *
 * CPP defines and macros for YAKL
 */

#pragma once
// Included by YAKL.h

#ifdef YAKL_B4B
  #define YAKL_MANAGED_MEMORY
#endif

#ifdef YAKL_DEBUG
  #ifndef YAKL_AUTO_FENCE
    #define YAKL_AUTO_FENCE
  #endif
#endif

#ifdef YAKL_AUTO_PROFILE
  #ifndef YAKL_PROFILE
    #define YAKL_PROFILE
  #endif
#endif

#ifdef YAKL_VERBOSE_FILE
  #ifndef YAKL_VERBOSE
    #define YAKL_VERBOSE
  #endif
#endif

#if defined(YAKL_AUTO_PROFILE) || defined(YAKL_VERBOSE)
  #define YAKL_AUTO_LABEL() (std::string(basename(__FILE__)) + std::string(":") + std::to_string(__LINE__)).c_str()
#else
  /** @brief This CPP Macro function will automatically create a label for you in the for of "filename: lineno".
    * @details Note that unless you have YAKL_AUTO_PROFILE or YAKL_VERBOSE defined, this returns an empty string
    *          for efficiency, since a label is not needed */
  #define YAKL_AUTO_LABEL() ""
#endif


#ifdef YAKL_ARCH_CUDA


  #define YAKL_LAMBDA [=] __host__ __device__
  #define YAKL_DEVICE_LAMBDA [=] __device__
  #define YAKL_CLASS_LAMBDA [=, *this] __host__ __device__
  #define YAKL_INLINE __host__ __device__ __forceinline__
  #define YAKL_DEVICE_INLINE __forceinline__ __device__
  #define YAKL_SCOPE(a,b) auto &a = b
  #ifndef YAKL_SINGLE_MEMORY_SPACE
    #define YAKL_SEPARATE_MEMORY_SPACE
  #endif
  #if defined(__NVCOMPILER_CUDA__)
    #define YAKL_EXECUTE_ON_HOST_ONLY(...) NV_IF_TARGET(NV_IS_HOST, __VA_ARGS__)
    #define YAKL_EXECUTE_ON_DEVICE_ONLY(...) NV_IF_TARGET(NV_IS_DEVICE, __VA_ARGS__)
  #else
    #ifdef __CUDA_ARCH__
      #define YAKL_EXECUTE_ON_HOST_ONLY(...)
      #define YAKL_EXECUTE_ON_DEVICE_ONLY(...) __VA_ARGS__
    #else
      #define YAKL_EXECUTE_ON_HOST_ONLY(...) __VA_ARGS__
      #define YAKL_EXECUTE_ON_DEVICE_ONLY(...) 
    #endif
  #endif


#elif defined(YAKL_ARCH_HIP)


  #define YAKL_LAMBDA [=] __host__ __device__
  #define YAKL_DEVICE_LAMBDA [=] __device__
  #define YAKL_CLASS_LAMBDA [=, *this] __host__ __device__
  #define YAKL_INLINE __host__ __device__ __forceinline__
  #define YAKL_DEVICE_INLINE __forceinline__ __device__
  #define YAKL_SCOPE(a,b) auto &a = std::ref(b).get()
  #ifndef YAKL_SINGLE_MEMORY_SPACE
    #define YAKL_SEPARATE_MEMORY_SPACE
  #endif
  #ifdef __HIP_DEVICE_COMPILE__
    #define YAKL_EXECUTE_ON_HOST_ONLY(...)
    #define YAKL_EXECUTE_ON_DEVICE_ONLY(...) __VA_ARGS__
  #else
    #define YAKL_EXECUTE_ON_HOST_ONLY(...) __VA_ARGS__
    #define YAKL_EXECUTE_ON_DEVICE_ONLY(...) 
  #endif


#elif defined(YAKL_ARCH_SYCL)


  #define YAKL_LAMBDA [=]
  #define YAKL_DEVICE_LAMBDA [=]
  #define YAKL_CLASS_LAMBDA [=, *this]
  #define YAKL_INLINE __inline__ __attribute__((always_inline))
  #define YAKL_DEVICE_INLINE __inline__ __attribute__((always_inline))
  #define YAKL_SCOPE(a,b) auto &a = std::ref(b).get()
  #ifndef YAKL_SINGLE_MEMORY_SPACE
    #define YAKL_SEPARATE_MEMORY_SPACE
  #endif
  #ifdef __SYCL_DEVICE_ONLY__
    #define YAKL_EXECUTE_ON_HOST_ONLY(...)
    #define YAKL_EXECUTE_ON_DEVICE_ONLY(...) __VA_ARGS__
    #define CL_CONSTANT __attribute__((opencl_constant))
  #else
    #define YAKL_EXECUTE_ON_HOST_ONLY(...) __VA_ARGS__
    #define YAKL_EXECUTE_ON_DEVICE_ONLY(...) 
    #define CL_CONSTANT
  #endif


#elif defined(YAKL_ARCH_OPENMP)


  #define YAKL_LAMBDA [=] 
  #define YAKL_DEVICE_LAMBDA [=] 
  #define YAKL_CLASS_LAMBDA [=, *this]
  #define YAKL_INLINE inline 
  #define YAKL_DEVICE_INLINE inline 
  #define YAKL_SCOPE(a,b) auto &a = b
  #define YAKL_EXECUTE_ON_HOST_ONLY(...) __VA_ARGS__
  #define YAKL_EXECUTE_ON_DEVICE_ONLY(...) __VA_ARGS__

#else

  /** @brief Used to create C++ lambda expressions passed to `parallel_for` and `parallel_outer`
    * @details This particular definition is for CPU targets only. It differs for other hardware backends. */
  #define YAKL_LAMBDA [=]

  /** @brief [NOT COMMONTLY USED] Used to create C++ lambda expressions only valid on the device
    * @details This particular definition is for CPU targets only. It differs for other hardware backends. */
  #define YAKL_DEVICE_LAMBDA [=]

  /** @brief Used to create C++ lambda expression that also pass `*this` by value in classes.
    * @details This particular definition is for CPU targets only. It differs for other hardware backends. */
  #define YAKL_CLASS_LAMBDA [=, *this]

  /** @brief Used to decorate functions called from kernels (`parallel_for` and `parallel_outer`) or from CPU functions.
    * @details This particular definition is for CPU targets only. It differs for other hardware backends. */
  #define YAKL_INLINE inline

  /** @brief [NOT COMMONLY USED] Used to decorate functions called **only** from kernels, not from CPU functions.
    * @details This particular definition is for CPU targets only. It differs for other hardware backends. */
  #define YAKL_DEVICE_INLINE inline

  /** @brief Used to bring non-local data into local scope (e.g., `this->data` or `namespace::data`).
    * Usage is, e.g., `YAKL_SCOPE(varname,this->varname);` or `YAKL_SCOPE(varname,::varname);`*/
  #define YAKL_SCOPE(a,b) auto &a = b

  /** @brief [NOT COMMONLY USED] Macro function used to determine if the current code is compiling for the host.
    * @details This particular definition is for CPU targets only. It differs for other hardware backends. 
    * This is used to hide device-only code from the host compiler. */
  #define YAKL_EXECUTE_ON_HOST_ONLY(...) __VA_ARGS__

  /** @brief [NOT COMMONLY USED] Macro function used to determine if the current code is compiling for the device.
    * @details This particular definition is for CPU targets only. It differs for other hardware backends. 
    * This is used to hide host-only code from the device compiler. */
  #define YAKL_EXECUTE_ON_DEVICE_ONLY(...) __VA_ARGS__

#endif

