
#pragma once



  // Initialize the YAKL framework
  inline void init( size_t poolBytes = 0 ) {

    std::function<void *( size_t )> alloc;
    std::function<void( void * )>   dealloc;

    #if   defined(__USE_CUDA__)
      #if defined (__MANAGED__)
        alloc   = [] ( size_t bytes ) -> void* {
          void *ptr;
          cudaMallocManaged(&ptr,bytes);
          cudaMemPrefetchAsync(ptr,bytes,0);
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
        };
      #else
        alloc   = [] ( size_t bytes ) -> void* {
          void *ptr;
          cudaMalloc(&ptr,bytes);
          return ptr;
        };
        dealloc = [] ( void *ptr    ) {
          cudaFree(ptr);
        };
      #endif
    #elif defined(__USE_HIP__)
      #if defined (__MANAGED__)
        alloc   = [] ( size_t bytes ) -> void* { void *ptr; hipMallocHost(&ptr,bytes); return ptr; };
        dealloc = [] ( void *ptr    )          { hipFree(ptr); };
      #else
        alloc   = [] ( size_t bytes ) -> void* { void *ptr; hipMalloc(&ptr,bytes); return ptr; };
        dealloc = [] ( void *ptr    )          { hipFree(ptr); };
      #endif
    #else
      alloc   = ::malloc;
      dealloc = ::free;
    #endif

    // If bytes are specified, then initialize a pool allocator
    if ( poolBytes > 0 ) {
      std::cout << "Initializing the YAKL Pool Allocator with " << poolBytes << " bytes" << std::endl;

      #if   defined(__USE_CUDA__)
        auto zerofunc = [] (void *ptr, size_t bytes) { cudaMemset(ptr,0,bytes); };
      #else
        auto zerofunc = [] (void *ptr, size_t bytes) { memset(ptr,0,bytes); };
      #endif

      pool = BuddyAllocator( poolBytes , 1024 , alloc , dealloc , zerofunc );

      yaklAllocDevice = [] (size_t bytes) -> void * { return pool.allocate( bytes ); };
      yaklFreeDevice  = [] (void *ptr)              { pool.free( ptr );              };

    } else { // poolBytes < 0
      std::cout << "Not using the YAKL Pool Allocator" << std::endl;

      yaklAllocDevice = alloc;
      yaklFreeDevice  = dealloc;

    } // poolBytes

    yaklAllocHost = [] (size_t bytes) -> void * { return malloc(bytes); };
    yaklFreeHost  = [] (void *ptr) { free(ptr); };

    #if defined(__USE_CUDA__)
      cudaMalloc(&functorBuffer,functorBufSize);
    #endif

    #if defined(__USE_HIP__)
      int id;
      hipGetDevice(&id);
      hipDeviceProp_t props;
      hipGetDeviceProperties(&props,id);
      std::cout << props.name << std::endl;
    #endif

    #if defined(__AUTO_FENCE__)
      std::cout << "WARNING: Automatically inserting fence() after every parallel_for" << std::endl;
    #endif

  } // 
