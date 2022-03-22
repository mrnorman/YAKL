
#pragma once
// Included by YAKL.h

namespace yakl {

  // Block the CPU code until the device code and data transfers are all completed
  inline void fence() {
    #if   defined(YAKL_ARCH_CUDA)
      cudaDeviceSynchronize();
    #elif defined(YAKL_ARCH_HIP)
      hipDeviceSynchronize();
    #elif defined(YAKL_ARCH_SYCL)
      sycl_default_stream().wait();
    #elif defined(YAKL_ARCH_OPENMP)
      #pragma omp barrier
    #endif
    check_last_error();
  }

  // Block further work on the inner parallelism level until previous work is completed
  YAKL_INLINE void fence_inner(InnerHandler &handler) {
    #if YAKL_CURRENTLY_ON_DEVICE()
      #if   defined(YAKL_ARCH_CUDA)
        __syncthreads();
      #elif defined(YAKL_ARCH_HIP)
        __syncthreads();
      #elif defined(YAKL_ARCH_SYCL)
        handler.get_item().barrier(sycl::access::fence_space::local_space);
      #elif defined(YAKL_ARCH_OPENMP)
        // OpenMP doesn't do parallelism at the inner level, so nothing needed here
      #endif
    #endif
  }

}


