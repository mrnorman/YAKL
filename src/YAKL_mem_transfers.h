
#pragma once

  template <class T1, class T2, typename std::enable_if< std::is_same< typename std::remove_cv<T1>::type ,
                                                                       typename std::remove_cv<T2>::type >::value , int >::type = 0>
  inline void memcpy_host_to_host(T1 *dst , T2 *src , index_t elems) {
    for (index_t i=0; i<elems; i++) { dst[i] = src[i]; }
  }


  template <class T1, class T2, typename std::enable_if< std::is_same< typename std::remove_cv<T1>::type ,
                                                                       typename std::remove_cv<T2>::type >::value , int >::type = 0>
  inline void memcpy_device_to_host(T1 *dst , T2 *src , index_t elems) {
    #ifdef YAKL_ARCH_CUDA
      cudaMemcpyAsync(dst,src,elems*sizeof(T1),cudaMemcpyDeviceToHost,0);
      check_last_error();
    #elif defined(YAKL_ARCH_HIP)
      hipMemcpyAsync(dst,src,elems*sizeof(T1),hipMemcpyDeviceToHost,0);
      check_last_error();
    #elif defined (YAKL_ARCH_SYCL)
      sycl_default_stream().memcpy(dst, src, elems*sizeof(T1));
      check_last_error();
    #elif defined(YAKL_ARCH_OPENMP45)
      omp_target_memcpy(dst,src,elems*sizeof(T1),0,0,omp_get_initial_device(),omp_get_default_device());
      check_last_error();
    #elif defined(YAKL_ARCH_OPENMP)
      #pragma omp parallel for
      for (index_t i=0; i<elems; i++) { dst[i] = src[i]; }
    #else
      for (index_t i=0; i<elems; i++) { dst[i] = src[i]; }
    #endif
    #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
      fence();
    #endif
  }


  template <class T1, class T2, typename std::enable_if< std::is_same< typename std::remove_cv<T1>::type ,
                                                                       typename std::remove_cv<T2>::type >::value , int >::type = 0>
  inline void memcpy_host_to_device(T1 *dst , T2 *src , index_t elems) {
    #ifdef YAKL_ARCH_CUDA
      cudaMemcpyAsync(dst,src,elems*sizeof(T1),cudaMemcpyHostToDevice,0);
      check_last_error();
    #elif defined(YAKL_ARCH_HIP)
      hipMemcpyAsync(dst,src,elems*sizeof(T1),hipMemcpyHostToDevice,0);
      check_last_error();
    #elif defined (YAKL_ARCH_SYCL)
      sycl_default_stream().memcpy(dst, src, elems*sizeof(T1));
      check_last_error();
    #elif defined(YAKL_ARCH_OPENMP45)
      omp_target_memcpy(dst,src,elems*sizeof(T1),0,0,omp_get_default_device(),omp_get_initial_device());
      check_last_error();
    #elif defined(YAKL_ARCH_OPENMP)
      #pragma omp parallel for
      for (index_t i=0; i<elems; i++) { dst[i] = src[i]; }
    #else
      for (index_t i=0; i<elems; i++) { dst[i] = src[i]; }
    #endif
    #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
      fence();
    #endif
  }


  template <class T1, class T2, typename std::enable_if< std::is_same< typename std::remove_cv<T1>::type ,
                                                                       typename std::remove_cv<T2>::type >::value , int >::type = 0>
  inline void memcpy_device_to_device(T1 *dst , T2 *src , index_t elems) {
    #ifdef YAKL_ARCH_CUDA
      cudaMemcpyAsync(dst,src,elems*sizeof(T1),cudaMemcpyDeviceToDevice,0);
      check_last_error();
    #elif defined(YAKL_ARCH_HIP)
      hipMemcpyAsync(dst,src,elems*sizeof(T1),hipMemcpyDeviceToDevice,0);
      check_last_error();
    #elif defined (YAKL_ARCH_SYCL)
      sycl_default_stream().memcpy(dst, src, elems*sizeof(T1));
      check_last_error();
    #elif defined(YAKL_ARCH_OPENMP45)
      omp_target_memcpy(dst,src,elems*sizeof(T1),0,0,omp_get_default_device(),omp_get_default_device());
      check_last_error();
    #elif defined(YAKL_ARCH_OPENMP)
      #pragma omp parallel for
      for (index_t i=0; i<elems; i++) { dst[i] = src[i]; }
    #else
      for (index_t i=0; i<elems; i++) { dst[i] = src[i]; }
    #endif
    #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
      fence();
    #endif
  }

