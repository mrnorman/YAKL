
#pragma once
// Included by YAKL.h

namespace yakl {

  // Set the allocation and deallocation functions for YAKL
  inline void set_alloc_free(std::function<void *( size_t )> &alloc , std::function<void ( void * )> &dealloc) {
    #if   defined(YAKL_ARCH_CUDA)
      #if defined (YAKL_MANAGED_MEMORY)
        alloc   = [] ( size_t bytes ) -> void* {
          if (bytes == 0) return nullptr;
          void *ptr;
          cudaMallocManaged(&ptr,bytes);      // Allocate managed memory
          check_last_error();
          #ifdef _OPENMP45
            // if using OMP target offload, make sure OMP runtime knows to leave this memory alone
            omp_target_associate_ptr(ptr,ptr,bytes,0,0);
          #endif
          #ifdef _OPENACC
            // if using OpenACC, make sure OpenACC runtime knows to leave this memory alone
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
          if (bytes == 0) return nullptr;
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
    #elif defined(YAKL_ARCH_HIP)
      #if defined (YAKL_MANAGED_MEMORY)
        alloc = [] ( size_t bytes ) -> void* {
          if (bytes == 0) return nullptr;
          void *ptr;
          hipMallocManaged(&ptr,bytes);  // This is the current standin for managed memory for HIP
          #ifdef _OPENMP45
            // if using OMP target offload, make sure OMP runtime knows to leave this memory alone
            omp_target_associate_ptr(ptr,ptr,bytes,0,0);
          #endif
          #ifdef _OPENACC
            // if using OpenACC, make sure OpenACC runtime knows to leave this memory alone
            acc_map_data(ptr,ptr,bytes);
          #endif
          check_last_error();
          return ptr;
        };
        dealloc = [] ( void *ptr    ) {
          hipFree(ptr);
          check_last_error();
        };
      #else
        alloc = [] ( size_t bytes ) -> void* {
          if (bytes == 0) return nullptr;
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
    #elif defined (YAKL_ARCH_SYCL)
      #if defined (YAKL_MANAGED_MEMORY)
        alloc = [] ( size_t bytes ) -> void* {
          if (bytes == 0) return nullptr;
          // Allocate unified shared memory
          void *ptr = sycl::malloc_shared(bytes,sycl_default_stream());
          sycl_default_stream().memset(ptr, 0, bytes);
          check_last_error();
          sycl_default_stream().prefetch(ptr,bytes);
          #ifdef _OPENMP45
            // if using OMP target offload, make sure OMP runtime knows to leave this memory alone
            omp_target_associate_ptr(ptr,ptr,bytes,0,0);
          #endif
          #ifdef _OPENACC
            // if using OpenACC, make sure OpenACC runtime knows to leave this memory alone
            acc_map_data(ptr,ptr,bytes);
          #endif
          return ptr;
        };
        dealloc = [] ( void *ptr ) {
          sycl::free(ptr, sycl_default_stream());
          check_last_error();
        };
      #else
        alloc = [] ( size_t bytes ) -> void* {
          if (bytes == 0) return nullptr;
          void *ptr = sycl::malloc_device(bytes,sycl_default_stream());
          sycl_default_stream().memset(ptr, 0, bytes);
          check_last_error();
          return ptr;
        };
        dealloc = [] ( void *ptr ) {
          sycl::free(ptr, sycl_default_stream());
          check_last_error();
          // ptr = nullptr;
        };
      #endif
    #else
      alloc   = [] ( size_t bytes ) -> void* { if (bytes == 0) return nullptr; return ::malloc(bytes); };
      dealloc = [] ( void *ptr ) { ::free(ptr); };
    #endif
  }

}

