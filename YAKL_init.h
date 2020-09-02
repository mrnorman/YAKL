
#pragma once



  // Initialize the YAKL framework
  inline void init() {
    bool use_pool = true;

    yakl_is_initialized = true;

    // Check for pool allocator env var
    char * env = std::getenv("GATOR_DISABLE");
    if ( env != nullptr ) {
      std::string resp(env);
      if (resp == "yes" || resp == "YES" || resp == "1" || resp == "true" || resp == "TRUE" || resp == "T") {
        use_pool = false;
      }
    }

    if (use_pool) {

      yaklAllocDeviceFunc = [] (size_t bytes , char const *label) -> void * {
        return pool.allocate( bytes , label );
      };
      yaklFreeDeviceFunc  = [] (void *ptr , char const *label)              {
        pool.free( ptr , label );
      };

    } else {

      std::function<void *( size_t , char const *)> alloc;
      std::function<void ( void * , char const *)>  dealloc;

      #if   defined(__USE_CUDA__)
        #if defined (__MANAGED__)
          alloc   = [] ( size_t bytes , char const *label ) -> void* {
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
          dealloc = [] ( void *ptr    , char const *label ) {
            cudaFree(ptr);
          };
        #else
          alloc   = [] ( size_t bytes , char const *label ) -> void* {
            void *ptr;
            cudaMalloc(&ptr,bytes);
            return ptr;
          };
          dealloc = [] ( void *ptr    , char const *label ) {
            cudaFree(ptr);
          };
        #endif
      #elif defined(__USE_HIP__)
        #if defined (__MANAGED__)
          alloc   = [] ( size_t bytes , char const *label ) -> void* { void *ptr; hipMallocHost(&ptr,bytes); return ptr; };
          dealloc = [] ( void *ptr    , char const *label )          { hipFree(ptr); };
        #else
          alloc   = [] ( size_t bytes , char const *label ) -> void* { void *ptr; hipMalloc(&ptr,bytes); return ptr; };
          dealloc = [] ( void *ptr    , char const *label )          { hipFree(ptr); };
        #endif
      #else
        alloc   = [] ( size_t bytes , char const *label ) -> void* { return ::malloc(bytes); };
        dealloc = [] ( void *ptr    , char const *label )          { ::free(ptr); };
      #endif

      yaklAllocDeviceFunc = alloc;
      yaklFreeDeviceFunc  = dealloc;
    }

    yaklAllocHostFunc = [] (size_t bytes , char const *label) -> void * { return malloc(bytes); };
    yaklFreeHostFunc  = [] (void *ptr , char const *label) { free(ptr); };

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
