
#pragma once

#ifdef YAKL_ARCH_CUDA
  #define YAKL_LAMBDA [=] __device__
  #define YAKL_INLINE inline __host__ __device__
  #define YAKL_DEVICE inline __device__
  #define YAKL_SCOPE(a,b) auto &a = b
  #include <cub/cub.cuh>
#elif defined(YAKL_ARCH_HIP)
  #define YAKL_LAMBDA [=] __host__ __device__
  #define YAKL_INLINE inline __host__ __device__
  #define YAKL_DEVICE inline __device__
  #define YAKL_SCOPE(a,b) auto &a = std::ref(b).get()
  #include "hip/hip_runtime.h"
  #include <hipcub/hipcub.hpp>
#elif defined(YAKL_ARCH_SYCL)
  #define YAKL_LAMBDA [=]
  #define YAKL_INLINE __inline__ __attribute__((always_inline))
  #define YAKL_DEVICE __inline__ __attribute__((always_inline))
  #define YAKL_SCOPE(a,b) auto &a = std::ref(b).get()
  #include <CL/sycl.hpp>
  namespace sycl = cl::sycl;
#elif defined(YAKL_ARCH_OPENMP45)
  #define YAKL_LAMBDA [=] 
  #define YAKL_INLINE inline 
  #define YAKL_DEVICE inline 
  #define YAKL_SCOPE(a,b) auto &a = std::ref(b).get()
  #include <omp.h>
#else
  #define YAKL_LAMBDA [=]
  #define YAKL_INLINE inline
  #define YAKL_DEVICE inline
  #define YAKL_SCOPE(a,b) auto &a = b
#endif

