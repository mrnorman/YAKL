
#pragma once
// Included by YAKL.h

namespace yakl {

  extern Gator pool;

  // YAKL allocator and deallocator on host and device as std::function's
  extern std::function<void *( size_t , char const *)> yaklAllocHost;
  extern std::function<void *( size_t , char const *)> yaklAllocDevice;
  extern std::function<void ( void * , char const *)>  yaklFreeHost;
  extern std::function<void ( void * , char const *)>  yaklFreeDevice;

  inline bool use_pool() {
    char * env = std::getenv("GATOR_DISABLE");
    if ( env != nullptr ) {
      std::string resp(env);
      if (resp == "yes" || resp == "YES" || resp == "1" || resp == "true" || resp == "TRUE" || resp == "T") {
        return false;
      }
    }
    return true;
  }


  // Set the allocation and deallocation functions for YAKL
  inline void set_device_alloc_free(std::function<void *( size_t )> &alloc , std::function<void ( void * )> &dealloc) {
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


  inline void set_yakl_allocators_to_default() {
    fence();
    if (use_pool()) {
      yaklAllocDevice = [] (size_t bytes , char const *label) -> void * { return pool.allocate( bytes , label ); };
      yaklFreeDevice  = [] (void *ptr , char const *label)              { pool.free( ptr , label ); };
    } else {
      std::function<void *( size_t)> alloc;
      std::function<void ( void *)>  dealloc;
      set_device_alloc_free(alloc , dealloc);
      yaklAllocDevice = [=] (size_t bytes , char const *label) -> void * { return alloc(bytes); };
      yaklFreeDevice  = [=] (void *ptr , char const *label)              { dealloc(ptr); };
    }
    yaklAllocHost = [] (size_t bytes , char const *label) -> void * { return malloc(bytes); };
    yaklFreeHost  = [] (void *ptr , char const *label) { free(ptr); };
  }


  inline void set_host_allocator    ( std::function<void *(size_t)> func ) {
    fence();   yaklAllocHost   = [=] (size_t bytes , char const *label) -> void * { return func(bytes); };
  }


  inline void set_device_allocator  ( std::function<void *(size_t)> func ) {
    fence();   yaklAllocDevice = [=] (size_t bytes , char const *label) -> void * { return func(bytes); };
  }


  inline void set_host_deallocator  ( std::function<void (void *)>  func ) {
    fence();   yaklFreeHost    = [=] (void *ptr , char const *label) { func(ptr); };
  }


  inline void set_device_deallocator( std::function<void (void *)>  func ) {
    fence();   yaklFreeDevice  = [=] (void *ptr , char const *label) { func(ptr); };
  }

  inline void set_host_allocator    ( std::function<void *( size_t , char const *)> func ) { fence();  yaklAllocHost   = func; }
  inline void set_device_allocator  ( std::function<void *( size_t , char const *)> func ) { fence();  yaklAllocDevice = func; }
  inline void set_host_deallocator  ( std::function<void ( void * , char const *)>  func ) { fence();  yaklFreeHost    = func; }
  inline void set_device_deallocator( std::function<void ( void * , char const *)>  func ) { fence();  yaklFreeDevice  = func; }

}


