
#pragma once

#ifdef __USE_CUDA__
  #define YAKL_LAMBDA [=] __device__
  #define YAKL_INLINE inline __host__ __device__
  #define YAKL_DEVICE inline __device__
  #define YAKL_SCOPE(a,b) auto &a = b
  #include <cub/cub.cuh>
#elif defined(__USE_HIP__)
  #define YAKL_LAMBDA [=] __host__ __device__
  #define YAKL_INLINE inline __host__ __device__
  #define YAKL_DEVICE inline __device__
  #define YAKL_SCOPE(a,b) auto &a = std::ref(b).get()
  #include "hip/hip_runtime.h"
  #include <hipcub/hipcub.hpp>
#elif defined(__USE_SYCL__)
  #define YAKL_LAMBDA [=]
  #define YAKL_INLINE __inline__ __attribute__((always_inline))
  #define YAKL_DEVICE __inline__ __attribute__((always_inline))
  #define YAKL_SCOPE(a,b) auto &a = std::ref(b).get()
  #include <CL/sycl.hpp>
  namespace sycl = cl::sycl;
#else
  #define YAKL_LAMBDA [=]
  #define YAKL_INLINE inline
  #define YAKL_DEVICE inline
  #define YAKL_SCOPE(a,b) auto &a = b
#endif

