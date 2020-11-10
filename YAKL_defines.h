
#pragma once

#ifdef __USE_CUDA__
  #define YAKL_LAMBDA [=] __device__
  #define YAKL_INLINE inline __host__ __device__
  #define YAKL_DEVICE inline __device__
  #include <cub/cub.cuh>
#elif defined(__USE_HIP__)
  #define YAKL_LAMBDA [=] __host__ __device__
  #define YAKL_INLINE inline __host__ __device__
  #define YAKL_DEVICE inline __device__
  #include "hip/hip_runtime.h"
  #include <hipcub/hipcub.hpp>
#elif defined(__USE_SYCL__)
  #define YAKL_LAMBDA [=]
  #define YAKL_INLINE inline
  #define YAKL_DEVICE inline
  #include <CL/sycl.hpp>
  namespace sycl = cl::sycl;
#else
  #define YAKL_LAMBDA [&]
  #define YAKL_INLINE inline
  #define YAKL_DEVICE inline
#endif

