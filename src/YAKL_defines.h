
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


#ifdef YAKL_ARCH_CUDA

  #define YAKL_LAMBDA [=] __host__ __device__
  #define YAKL_DEVICE_LAMBDA [=] __device__
  #define YAKL_INLINE __host__ __device__ __forceinline__
  #define YAKL_DEVICE_INLINE __forceinline__ __device__
  #define YAKL_SCOPE(a,b) auto &a = b
  #ifndef YAKL_SINGLE_MEMORY_SPACE
    #define YAKL_SEPARATE_MEMORY_SPACE
  #endif
  #define YAKL_CURRENTLY_ON_HOST() (! defined(__CUDA_ARCH__))
  #define YAKL_CURRENTLY_ON_DEVICE() (defined(__CUDA_ARCH__))

#elif defined(YAKL_ARCH_HIP)

  #define YAKL_LAMBDA [=] __host__ __device__
  #define YAKL_DEVICE_LAMBDA [=] __device__
  #define YAKL_INLINE __host__ __device__ __forceinline__
  #define YAKL_DEVICE_INLINE __forceinline__ __device__
  #define YAKL_SCOPE(a,b) auto &a = std::ref(b).get()
  #ifndef YAKL_SINGLE_MEMORY_SPACE
    #define YAKL_SEPARATE_MEMORY_SPACE
  #endif
  #define YAKL_CURRENTLY_ON_HOST() (! defined(__HIP_DEVICE_COMPILE__))
  #define YAKL_CURRENTLY_ON_DEVICE() (defined(__HIP_DEVICE_COMPILE__))

#elif defined(YAKL_ARCH_SYCL)

  #define YAKL_LAMBDA [=]
  #define YAKL_DEVICE_LAMBDA [=]
  #define YAKL_INLINE __inline__ __attribute__((always_inline))
  #define YAKL_DEVICE_INLINE __inline__ __attribute__((always_inline))
  #define YAKL_SCOPE(a,b) auto &a = std::ref(b).get()
  #ifndef YAKL_SINGLE_MEMORY_SPACE
    #define YAKL_SEPARATE_MEMORY_SPACE
  #endif
  #define YAKL_CURRENTLY_ON_HOST() (! defined(__SYCL_DEVICE_ONLY__))
  #define YAKL_CURRENTLY_ON_DEVICE() (defined(__SYCL_DEVICE_ONLY__))
  #ifdef __SYCL_DEVICE_ONLY__
    #define CL_CONSTANT __attribute__((opencl_constant))
  #else
    #define CL_CONSTANT
  #endif

#elif defined(YAKL_ARCH_OPENMP)

  #define YAKL_LAMBDA [=] 
  #define YAKL_DEVICE_LAMBDA [=] 
  #define YAKL_INLINE inline 
  #define YAKL_DEVICE_INLINE inline 
  #define YAKL_SCOPE(a,b) auto &a = b
  #define YAKL_CURRENTLY_ON_HOST() 1
  #define YAKL_CURRENTLY_ON_DEVICE() 1

#else

  #define YAKL_LAMBDA [=]
  #define YAKL_DEVICE_LAMBDA [=]
  #define YAKL_INLINE inline
  #define YAKL_DEVICE_INLINE inline
  #define YAKL_SCOPE(a,b) auto &a = b
  #define YAKL_CURRENTLY_ON_HOST() 1
  #define YAKL_CURRENTLY_ON_DEVICE() 1

#endif

