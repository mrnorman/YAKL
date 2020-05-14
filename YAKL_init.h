
#pragma once



  // Initialize the YAKL framework
  inline void init() {
    yaklAllocDeviceFunc = [] (size_t bytes) -> void * { return pool.allocate( bytes ); };
    yaklFreeDeviceFunc  = [] (void *ptr)              { pool.free( ptr );              };

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
