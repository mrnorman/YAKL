
#pragma once

#ifdef YAKL_ARCH_CUDA

  #define YAKL_LAMBDA [=] __host__ __device__
  #define YAKL_DEVICE_LAMBDA [=] __device__
  #define YAKL_INLINE inline __host__ __device__
  #define YAKL_DEVICE_INLINE inline __device__
  #define YAKL_SCOPE(a,b) auto &a = b
  #define YAKL_SEPARATE_MEMORY_SPACE
  #include <cub/cub.cuh>

#elif defined(YAKL_ARCH_HIP)

  #define YAKL_LAMBDA [=] __host__ __device__
  #define YAKL_DEVICE_LAMBDA [=] __device__
  #define YAKL_INLINE inline __host__ __device__
  #define YAKL_DEVICE_INLINE inline __device__
  #define YAKL_SCOPE(a,b) auto &a = std::ref(b).get()
  #define YAKL_SEPARATE_MEMORY_SPACE
  #include "hip/hip_runtime.h"
  #include <hipcub/hipcub.hpp>

#elif defined(YAKL_ARCH_SYCL)

  #define YAKL_LAMBDA [=]
  #define YAKL_DEVICE_LAMBDA [=]
  #define YAKL_INLINE __inline__ __attribute__((always_inline))
  #define YAKL_DEVICE_INLINE __inline__ __attribute__((always_inline))
  #define YAKL_SCOPE(a,b) auto &a = std::ref(b).get()
  #define YAKL_SEPARATE_MEMORY_SPACE
  #include <CL/sycl.hpp>
  namespace sycl = cl::sycl;

#elif defined(YAKL_ARCH_OPENMP45)

  #define YAKL_LAMBDA [=] 
  #define YAKL_DEVICE_LAMBDA [=] 
  #define YAKL_INLINE inline 
  #define YAKL_DEVICE_INLINE inline 
  #define YAKL_SCOPE(a,b) auto &a = std::ref(b).get()
  #define YAKL_SEPARATE_MEMORY_SPACE
  #include <omp.h>

#elif defined(YAKL_ARCH_OPENMP)

  #define YAKL_LAMBDA [=] 
  #define YAKL_DEVICE_LAMBDA [=] 
  #define YAKL_INLINE inline 
  #define YAKL_DEVICE_INLINE inline 
  #define YAKL_SCOPE(a,b) auto &a = b

#else

  #define YAKL_LAMBDA [=]
  #define YAKL_DEVICE_LAMBDA [=]
  #define YAKL_INLINE inline
  #define YAKL_DEVICE_INLINE inline
  #define YAKL_SCOPE(a,b) auto &a = b

#endif

