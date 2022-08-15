/**
 * @file
 * YAKL memory transfoer routines
 */

#pragma once
// Included by YAKL.h

namespace yakl {

  // Your one-stop shop for memory transfers to / from host / device

  /**
   * @brief [USE AT YOUR OWN RISK]: memcpy the specified number of **elements** on the host
   */
  template <class T1, class T2,
            typename std::enable_if< std::is_same< typename std::remove_cv<T1>::type ,
                                                   typename std::remove_cv<T2>::type >::value , int >::type = 0>
  inline void memcpy_host_to_host(T1 *dst , T2 *src , index_t elems) {
    #ifdef YAKL_AUTO_PROFILE
      timer_start("YAKL_internal_memcpy_host_to_host");
    #endif
    for (index_t i=0; i<elems; i++) { dst[i] = src[i]; }
    #ifdef YAKL_AUTO_PROFILE
      timer_stop("YAKL_internal_memcpy_host_to_host");
    #endif
  }


  /**
   * @brief [USE AT YOUR OWN RISK]: memcpy the specified number of **bytes** on the host
   */
  inline void memcpy_host_to_host_void(void *dst , void *src , size_t bytes) {
    #ifdef YAKL_AUTO_PROFILE
      timer_start("YAKL_internal_memcpy_host_to_host");
    #endif
    memcpy( dst , src , bytes );
    #ifdef YAKL_AUTO_PROFILE
      timer_stop("YAKL_internal_memcpy_host_to_host");
    #endif
  }


  /**
   * @brief [USE AT YOUR OWN RISK]: memcpy the specified number of **elements** from device to host
   */
  template <class T1, class T2,
            typename std::enable_if< std::is_same< typename std::remove_cv<T1>::type ,
                                                   typename std::remove_cv<T2>::type >::value , int >::type = 0>
  inline void memcpy_device_to_host(T1 *dst , T2 *src , index_t elems , Stream stream = Stream() ) {
    #ifdef YAKL_AUTO_PROFILE
      timer_start("YAKL_internal_memcpy_device_to_host");
    #endif
    #ifdef YAKL_ARCH_CUDA
      cudaMemcpyAsync(dst,src,elems*sizeof(T1),cudaMemcpyDeviceToHost,stream.get_real_stream());
      check_last_error();
    #elif defined(YAKL_ARCH_HIP)
      hipMemcpyAsync(dst,src,elems*sizeof(T1),hipMemcpyDeviceToHost,stream.get_real_stream());
      check_last_error();
    #elif defined (YAKL_ARCH_SYCL)
      stream.get_real_stream().memcpy(dst, src, elems*sizeof(T1));
      check_last_error();
    #elif defined(YAKL_ARCH_OPENMP)
      #pragma omp parallel for
      for (index_t i=0; i<elems; i++) { dst[i] = src[i]; }
    #else
      for (index_t i=0; i<elems; i++) { dst[i] = src[i]; }
    #endif
    #if defined(YAKL_AUTO_FENCE)
      fence();
    #endif
    #ifdef YAKL_AUTO_PROFILE
      timer_stop("YAKL_internal_memcpy_device_to_host");
    #endif
  }


  /**
   * @brief [USE AT YOUR OWN RISK]: memcpy the specified number of **elements** from host to device
   */
  template <class T1, class T2,
            typename std::enable_if< std::is_same< typename std::remove_cv<T1>::type ,
                                                   typename std::remove_cv<T2>::type >::value , int >::type = 0>
  inline void memcpy_host_to_device(T1 *dst , T2 *src , index_t elems , Stream stream = Stream() ) {
    #ifdef YAKL_AUTO_PROFILE
      timer_start("YAKL_internal_memcpy_host_to_device");
    #endif
    #ifdef YAKL_ARCH_CUDA
      cudaMemcpyAsync(dst,src,elems*sizeof(T1),cudaMemcpyHostToDevice,stream.get_real_stream());
      check_last_error();
    #elif defined(YAKL_ARCH_HIP)
      hipMemcpyAsync(dst,src,elems*sizeof(T1),hipMemcpyHostToDevice,stream.get_real_stream());
      check_last_error();
    #elif defined (YAKL_ARCH_SYCL)
      stream.get_real_stream().memcpy(dst, src, elems*sizeof(T1));
      check_last_error();
    #elif defined(YAKL_ARCH_OPENMP)
      #pragma omp parallel for
      for (index_t i=0; i<elems; i++) { dst[i] = src[i]; }
    #else
      for (index_t i=0; i<elems; i++) { dst[i] = src[i]; }
    #endif
    #if defined(YAKL_AUTO_FENCE)
      fence();
    #endif
    #ifdef YAKL_AUTO_PROFILE
      timer_stop("YAKL_internal_memcpy_host_to_device");
    #endif
  }


  /**
   * @brief [USE AT YOUR OWN RISK]: memcpy the specified number of **elements** on the device
   */
  template <class T1, class T2,
            typename std::enable_if< std::is_same< typename std::remove_cv<T1>::type ,
                                                   typename std::remove_cv<T2>::type >::value , int >::type = 0>
  inline void memcpy_device_to_device(T1 *dst , T2 *src , index_t elems , Stream stream = Stream() ) {
    #ifdef YAKL_AUTO_PROFILE
      timer_start("YAKL_internal_memcpy_device_to_device");
    #endif
    #ifdef YAKL_ARCH_CUDA
      cudaMemcpyAsync(dst,src,elems*sizeof(T1),cudaMemcpyDeviceToDevice,stream.get_real_stream());
      check_last_error();
    #elif defined(YAKL_ARCH_HIP)
      hipMemcpyAsync(dst,src,elems*sizeof(T1),hipMemcpyDeviceToDevice,stream.get_real_stream());
      check_last_error();
    #elif defined (YAKL_ARCH_SYCL)
      stream.get_real_stream().memcpy(dst, src, elems*sizeof(T1));
      check_last_error();
    #elif defined(YAKL_ARCH_OPENMP)
      #pragma omp parallel for
      for (index_t i=0; i<elems; i++) { dst[i] = src[i]; }
    #else
      for (index_t i=0; i<elems; i++) { dst[i] = src[i]; }
    #endif
    #if defined(YAKL_AUTO_FENCE)
      fence();
    #endif
    #ifdef YAKL_AUTO_PROFILE
      timer_stop("YAKL_internal_memcpy_device_to_device");
    #endif
  }


  /**
   * @brief [USE AT YOUR OWN RISK]: memcpy the specified number of **bytes** on the device
   */
  inline void memcpy_device_to_device_void(void *dst , void *src , size_t bytes , Stream stream = Stream() ) {
    #ifdef YAKL_AUTO_PROFILE
      timer_start("YAKL_internal_memcpy_device_to_device");
    #endif
    #ifdef YAKL_ARCH_CUDA
      cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToDevice,stream.get_real_stream());
      check_last_error();
    #elif defined(YAKL_ARCH_HIP)
      hipMemcpyAsync(dst,src,bytes,hipMemcpyDeviceToDevice,stream.get_real_stream());
      check_last_error();
    #elif defined (YAKL_ARCH_SYCL)
      stream.get_real_stream().memcpy(dst, src, bytes);
      check_last_error();
    #else
      memcpy( dst , src , bytes );
    #endif
    #if defined(YAKL_AUTO_FENCE)
      fence();
    #endif
    #ifdef YAKL_AUTO_PROFILE
      timer_stop("YAKL_internal_memcpy_device_to_device");
    #endif
  }

}


