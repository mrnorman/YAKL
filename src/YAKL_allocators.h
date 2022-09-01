/**
 * @file
 *
 * Contains functions related to controlling YAKL's allocators and deallocators.
 */

#pragma once
// Included by YAKL.h

namespace yakl {

  extern Gator pool;

  // YAKL allocator and deallocator device as std::function's
  extern std::function<void *( size_t , char const *)> alloc_device_func;
  extern std::function<void ( void * , char const *)>  free_device_func; 

  extern bool device_allocators_are_default;
  extern bool pool_enabled;

  /**
   * @brief If true, then the pool allocator is being used for all device allocations
   */
  inline bool use_pool() { return pool_enabled; }


  // Set the allocation and deallocation functions for YAKL
  /** @private */
  inline void set_device_alloc_free(std::function<void *( size_t )> &alloc , std::function<void ( void * )> &dealloc) {
    #if   defined(YAKL_ARCH_CUDA)
      #if defined (YAKL_MANAGED_MEMORY)
        alloc   = [] ( size_t bytes ) -> void* {
          if (bytes == 0) return nullptr;
          void *ptr;
          cudaMallocManaged(&ptr,bytes);      // Allocate managed memory
          if (ptr == nullptr) yakl::yakl_throw("ERROR: cudaMallocManaged returned nullptr. You have likely run out of memory");
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
          cudaFree(ptr);
          check_last_error();
        };
      #else
        alloc   = [] ( size_t bytes ) -> void* {
          if (bytes == 0) return nullptr;
          void *ptr;
          cudaMalloc(&ptr,bytes);
          if (ptr == nullptr) yakl::yakl_throw("ERROR: cudaMalloc returned nullptr. You have likely run out of memory");
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
          if (ptr == nullptr) yakl::yakl_throw("ERROR: hipMallocManaged returned nullptr. You have likely run out of memory");
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
          if (ptr == nullptr) yakl::yakl_throw("ERROR: hipMalloc returned nullptr. You have likely run out of memory");
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
          if (ptr == nullptr) yakl::yakl_throw("ERROR: sycl::malloc_shared returned nullptr. You have likely run out of memory");
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
          #ifdef YAKL_ENABLE_STREAMS
            fence();
          #endif
          void *ptr = sycl::malloc_device(bytes,sycl_default_stream());
          #ifdef YAKL_ENABLE_STREAMS
            fence();
          #endif
          if (ptr == nullptr) yakl::yakl_throw("ERROR: sycl::malloc_device returned nullptr. You have likely run out of memory");
          #ifdef YAKL_ENABLE_STREAMS
            fence();
          #endif
          sycl_default_stream().memset(ptr, 0, bytes);
          #ifdef YAKL_ENABLE_STREAMS
            fence();
          #endif
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
      alloc   = [] ( size_t bytes ) -> void* {
        if (bytes == 0) return nullptr;
        void *ptr = ::malloc(bytes);
        if (ptr == nullptr) yakl::yakl_throw("ERROR: malloc returned nullptr. You have likely run out of memory");
        return ptr;
      };
      dealloc = [] ( void *ptr ) {
        ::free(ptr);
      };
    #endif
  }


  /**
   * @brief Return all YAKL allocators to their defaults.
   *        
   * If the user has not overridden YAKL's default allocators, then this has no effect.
   */
  inline void set_yakl_allocators_to_default() {
    fence();
    if (use_pool()) {
      alloc_device_func = [] (size_t bytes , char const *label) -> void * {
        #ifdef YAKL_MEMORY_DEBUG
          if (yakl_mainproc()) std::cout << "MEMORY_DEBUG: Allocating label \"" << label << "\" of size " << bytes << " bytes" << std::endl;
        #endif
        void * ptr = pool.allocate( bytes , label );
        #ifdef YAKL_MEMORY_DEBUG
          if (yakl_mainproc()) std::cout << "MEMORY_DEBUG: Successfully allocated label \"" << label
                                         << " with pointer address " << ptr << std::endl;
        #endif
        return ptr;
      };
      free_device_func  = [] (void *ptr , char const *label)              {
        #ifdef YAKL_MEMORY_DEBUG
          if (yakl_mainproc()) std::cout << "MEMORY_DEBUG: Freeing label \"" << label << "\" with pointer address " << ptr << std::endl;
        #endif
        pool.free( ptr , label );
      };
    } else {
      std::function<void *( size_t)> alloc;
      std::function<void ( void *)>  dealloc;
      set_device_alloc_free(alloc , dealloc);
      alloc_device_func = [=] (size_t bytes , char const *label) -> void * {
        #ifdef YAKL_MEMORY_DEBUG
          if (yakl_mainproc()) std::cout << "MEMORY_DEBUG: Allocating label \"" << label << "\" of size " << bytes << " bytes" << std::endl;
        #endif
        void * ptr = alloc(bytes);
        #ifdef YAKL_MEMORY_DEBUG
          if (yakl_mainproc()) std::cout << "MEMORY_DEBUG: Successfully allocated label \"" << label
                                         << " with pointer address " << ptr << std::endl;
        #endif
        return ptr;
      };
      free_device_func  = [=] (void *ptr , char const *label)              {
        #ifdef YAKL_MEMORY_DEBUG
          if (yakl_mainproc()) std::cout << "MEMORY_DEBUG: Freeing label \"" << label << "\" with pointer address " << ptr << std::endl;
        #endif
        dealloc(ptr);
      };
    }

    device_allocators_are_default = true;
  }


  /**
   * @brief Override YAKL's device allocator with the passed function (No Label).
   * 
   * After overriding one of YAKL's allocators or deallocators, the passed function will be used until the user
   * overrides it again or calls yakl::set_yakl_allocators_to_default(). There are overriding functions that accept
   * labels for bookkeeping and debugging, and there are functions that do not use labels.
   */
  inline void set_device_allocator  ( std::function<void *(size_t)> func ) {
    fence();   alloc_device_func = [=] (size_t bytes , char const *label) -> void * { return func(bytes); };
    device_allocators_are_default = false;
  }


  /**
   * @brief Override YAKL's device deallocator with the passed function (No Label).
   * \copydetails yakl::set_device_allocator
   */
  inline void set_device_deallocator( std::function<void (void *)>  func ) {
    fence();   free_device_func  = [=] (void *ptr , char const *label) { func(ptr); };
    device_allocators_are_default = false;
  }


  /**
   * @brief Override YAKL's device allocator with the passed function (WITH Label).
   * \copydetails yakl::set_device_allocator
   */
  inline void set_device_allocator  ( std::function<void *( size_t , char const *)> func ) { fence();  alloc_device_func = func; }


  /**
   * @brief Override YAKL's device deallocator with the passed function (WITH Label).
   * \copydetails yakl::set_device_allocator
   */
  inline void set_device_deallocator( std::function<void ( void * , char const *)>  func ) { fence();  free_device_func  = func; }

  /** @brief Allocate on the device using YAKL's device allocator */
  inline void * alloc_device( size_t bytes, char const *label) { return alloc_device_func(bytes,label); }

  /** @brief Free on the device using YAKL's device deallocator */
  inline void   free_device ( void * ptr  , char const *label) {        free_device_func (ptr  ,label); }

}


