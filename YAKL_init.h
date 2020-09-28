
#pragma once

#include "YAKL_alloc_free.h"


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

      std::function<void *( size_t)> alloc;
      std::function<void ( void *)>  dealloc;
      set_alloc_free(alloc , dealloc);
      yaklAllocDeviceFunc = [&] (size_t bytes , char const *label) -> void * { return alloc(bytes); };
      yaklFreeDeviceFunc  = [&] (void *ptr , char const *label)              { dealloc(ptr); };
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
