
#pragma once

template <class T> constexpr T fastmod(T a , T b) {
  return a < b ? a : a-b*(a/b);
}

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


template <int VL = YAKL_DEFAULT_VECTOR_LEN>
struct LaunchConfig { };


#include "YAKL_parallel_for_c.h"

#include "YAKL_parallel_for_fortran.h"



