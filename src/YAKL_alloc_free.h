
#pragma once

inline void set_alloc_free(std::function<void *( size_t )> &alloc , std::function<void ( void * )> &dealloc) {
  #if   defined(YAKL_ARCH_CUDA)
    #if defined (YAKL_MANAGED_MEMORY)
      alloc   = [] ( size_t bytes ) -> void* {
        if (bytes == 0) return nullptr;
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
        void *ptr = sycl::malloc_shared(bytes,sycl_default_stream());
        sycl_default_stream().memset(ptr, 0, bytes);
        check_last_error();
        sycl_default_stream().prefetch(ptr,bytes);
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
  #elif defined(YAKL_ARCH_OPENMP45)
    alloc = [] ( size_t bytes ) -> void* {
      if (bytes == 0) return nullptr;
      void *ptr;
      int device;
      device = omp_get_default_device();
      ptr = omp_target_alloc(bytes,device);
      //check does nothing
      check_last_error();
      return ptr;
    };
    dealloc = [] (void *ptr) {
      int device;
      device = omp_get_default_device();
      omp_target_free(ptr,device);
      //check does nothing
      check_last_error();
    };
  #else
    alloc   = [] ( size_t bytes ) -> void* { if (bytes == 0) return nullptr; return ::malloc(bytes); };
    dealloc = [] ( void *ptr ) { ::free(ptr); };
  #endif
}

