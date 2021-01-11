
#pragma once

#include "YAKL_error.h"

namespace yakl {

  inline void set_alloc_free(std::function<void *( size_t )> &alloc , std::function<void ( void * )> &dealloc) {
    #if   defined(__USE_CUDA__)
      #if defined (__MANAGED__)
        alloc   = [] ( size_t bytes ) -> void* {
          void *ptr;
          cudaMallocManaged(&ptr,bytes);
          check_last_error();
          cudaMemPrefetchAsync(ptr,bytes,0);
          check_last_error();
          #ifdef _OPENMP45
            omp_target_associate_ptr(ptr,ptr,bytes,0,0);
          #endif
          #ifdef _OPENACC
            acc_map_data(ptr,ptr,bytes);
          #endif
          return ptr;
        };
        dealloc = [] ( void *ptr    ) {
          cudaFree(ptr);
          check_last_error();
        };
      #else
        alloc   = [] ( size_t bytes ) -> void* {
          void *ptr;
          cudaMalloc(&ptr,bytes);
          check_last_error();
          return ptr;
        };
        dealloc = [] ( void *ptr    ) {
          cudaFree(ptr);
          check_last_error();
        };
      #endif
    #elif defined(__USE_HIP__)
      #if defined (__MANAGED__)
        alloc = [] ( size_t bytes ) -> void* {
          void *ptr;
          hipMallocHost(&ptr,bytes);
          check_last_error();
          return ptr;
        };
        dealloc = [] ( void *ptr    ) {
          hipFree(ptr);
          check_last_error();
        };
      #else
        alloc = [] ( size_t bytes ) -> void* {
          void *ptr;
          hipMalloc(&ptr,bytes);
          check_last_error();
          return ptr;
        };
        dealloc = [] ( void *ptr ) {
          hipFree(ptr);
          check_last_error();
        };
      #endif
    #elif defined (__USE_SYCL__)
        alloc = [] ( size_t bytes ) -> void* {
          void *ptr = static_cast<int *>(sycl::malloc_device(bytes,sycl_default_stream));
          check_last_error();
          return ptr;
        };
        dealloc = [] ( void *ptr ) {
          sycl::free(ptr, sycl_default_stream);
          check_last_error();
        };
    #elif defined(__USE_OPENMP45__)
      alloc = [] ( size_t bytes ) -> void* {
        int device = omp_get_default_device();
        std::cout << device << "\n";
        void *ptr = omp_target_alloc(bytes,device);
        //check does nothing
        check_last_error();
        return ptr;
      };
      dealloc = [] (void *ptr) {
        int device = omp_get_default_device();
        omp_target_free(ptr,device);
        //check does nothing
        check_last_error();
      };
    #else
      alloc   = [] ( size_t bytes ) -> void* { return ::malloc(bytes); };
      dealloc = [] ( void *ptr ) { ::free(ptr); };
    #endif
  }

}

