/**
 * @file
 * YAKL fence routines to block code are varying levels until all threads / streams
 * have completed on the device.
 */

#pragma once
// Included by YAKL.h

__YAKL_NAMESPACE_WRAPPER_BEGIN__
namespace yakl {

  /**
   * @brief Block the host code until all device code has completed.
   */
  inline void fence() {
    #if   defined(YAKL_ARCH_CUDA)
      cudaDeviceSynchronize();
    #elif defined(YAKL_ARCH_HIP)
      hipDeviceSynchronize();
    #elif defined(YAKL_ARCH_SYCL)
      sycl_default_stream().wait_and_throw();
    #elif defined(YAKL_ARCH_OPENMP)
      #pragma omp barrier
    #endif
  }

  /**
   * @brief Block inner threads until all inner threads have completed.
   * @details To be called inside yakl::parallel_outer *only*. Block the inner-level parallelism
   * until all inner threads have reached this point. In CUDA and HIP, this is __syncthreads(). 
   * @param handler The yakl::InnerHandler object create by yakl::parallel_outer
   */
  YAKL_INLINE void fence_inner(InnerHandler &handler) {
    #if   defined(YAKL_ARCH_CUDA)
      YAKL_EXECUTE_ON_DEVICE_ONLY( __syncthreads(); )
    #elif defined(YAKL_ARCH_HIP)
      YAKL_EXECUTE_ON_DEVICE_ONLY( __syncthreads(); )
    #elif defined(YAKL_ARCH_SYCL)
      YAKL_EXECUTE_ON_DEVICE_ONLY( handler.get_item().barrier(sycl::access::fence_space::local_space); )
    #elif defined(YAKL_ARCH_OPENMP)
      // OpenMP doesn't do parallelism at the inner level, so nothing needed here
    #endif
  }

}
__YAKL_NAMESPACE_WRAPPER_END__


