
#pragma once
// Included by YAKL.h

namespace yakl {

  // Block the CPU code until the device code and data transfers are all completed
  inline void fence() {
    #ifdef YAKL_ARCH_CUDA
      cudaDeviceSynchronize();
      check_last_error();
    #endif
    #ifdef YAKL_ARCH_HIP
      hipDeviceSynchronize();
      check_last_error();
    #endif
    #ifdef YAKL_ARCH_SYCL
      sycl_default_stream().wait();
      check_last_error();
    #endif
  }

  // Block the CPU code until the device code and data transfers are all completed
  YAKL_INLINE void fence_inner() {
    #if YAKL_CURRENTLY_ON_DEVICE()
      #ifdef YAKL_ARCH_CUDA
        __syncthreads();
      #endif
      #ifdef YAKL_ARCH_HIP
        __syncthreads();
      #endif
      #ifdef YAKL_ARCH_SYCL
        sycl_default_stream().wait();
        check_last_error();
      #endif
    #endif
  }

}


