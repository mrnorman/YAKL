
#pragma once

  template <class T>
  inline void memcpy_host_to_host(T *dst , T *src , index_t elems) {
    for (index_t i=0; i<elems; i++) { dst[i] = src[i]; }
  }


  template <class T>
  inline void memcpy_device_to_host(T *dst , T *src , index_t elems) {
    #ifdef YAKL_ARCH_CUDA
      cudaMemcpyAsync(dst,src,elems*sizeof(T),cudaMemcpyDeviceToHost,0);
      check_last_error();
    #elif defined(YAKL_ARCH_HIP)
      hipMemcpyAsync(dst,src,elems*sizeof(T),hipMemcpyDeviceToHost,0);
      check_last_error();
    #elif defined (YAKL_ARCH_SYCL)
      sycl_default_stream.memcpy(dst, src, elems*sizeof(T));
      check_last_error();
    #elif defined(YAKL_ARCH_OPENMP45)
      omp_target_memcpy(dst,src,elems*sizeof(T),0,0,omp_get_initial_device(),omp_get_default_device());
      check_last_error();
    #else
      for (index_t i=0; i<elems; i++) { dst[i] = src[i]; }
    #endif
    #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
      fence();
    #endif
  }


  template <class T>
  inline void memcpy_host_to_device(T *dst , T *src , index_t elems) {
    #ifdef YAKL_ARCH_CUDA
      cudaMemcpyAsync(dst,src,elems*sizeof(T),cudaMemcpyHostToDevice,0);
      check_last_error();
    #elif defined(YAKL_ARCH_HIP)
      hipMemcpyAsync(dst,src,elems*sizeof(T),hipMemcpyHostToDevice,0);
      check_last_error();
    #elif defined (YAKL_ARCH_SYCL)
      sycl_default_stream.memcpy(dst, src, elems*sizeof(T));
      check_last_error();
    #elif defined(YAKL_ARCH_OPENMP45)
      omp_target_memcpy(dst,src,elems*sizeof(T),0,0,omp_get_default_device(),omp_get_initial_device());
      check_last_error();
    #else
      for (index_t i=0; i<elems; i++) { dst[i] = src[i]; }
    #endif
    #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
      fence();
    #endif
  }


  template <class T>
  inline void memcpy_device_to_device(T *dst , T *src , index_t elems) {
    #ifdef YAKL_ARCH_CUDA
      cudaMemcpyAsync(dst,src,elems*sizeof(T),cudaMemcpyDeviceToDevice,0);
      check_last_error();
    #elif defined(YAKL_ARCH_HIP)
      hipMemcpyAsync(dst,src,elems*sizeof(T),hipMemcpyDeviceToDevice,0);
      check_last_error();
    #elif defined (YAKL_ARCH_SYCL)
      sycl_default_stream.memcpy(dst, src, elems*sizeof(T));
      check_last_error();
    #elif defined(YAKL_ARCH_OPENMP45)
      omp_target_memcpy(dst,src,elems*sizeof(T),0,0,omp_get_default_device(),omp_get_default_device());
      check_last_error();
    #else
      for (index_t i=0; i<elems; i++) { dst[i] = src[i]; }
    #endif
    #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
      fence();
    #endif
  }

