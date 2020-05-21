
#pragma once



  // Initialize the YAKL framework
  inline void init() {
    bool use_pool = true;

    // Check for pool allocator env var
    char * env = std::getenv("GATOR_DISABLE");
    if ( env != nullptr ) {
      std::string resp(env);
      if (resp == "yes" || resp == "YES" || resp == "1" || resp == "true" || resp == "TRUE" || resp == "T") {
        use_pool = false;
      }
    }

    if (use_pool) {

      yaklAllocDeviceFunc = [] (size_t bytes) -> void * { return pool.allocate( bytes ); };
      yaklFreeDeviceFunc  = [] (void *ptr)              { pool.free( ptr );              };

    } else {

      std::function<void *( size_t )> alloc;
      std::function<void ( void * )>  dealloc;

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

      yaklAllocDeviceFunc = alloc;
      yaklFreeDeviceFunc  = dealloc;
    }

    yaklAllocHostFunc = [] (size_t bytes) -> void * { return malloc(bytes); };
    yaklFreeHostFunc  = [] (void *ptr) { free(ptr); };

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
