
#pragma once

#include "YAKL_alloc_free.h"

  // Initialize the YAKL framework
  inline void init() {
    yakl_mtx.lock();

    if ( ! isInitialized() ) {
      #if defined(YAKL_PROFILE) || defined(YAKL_AUTO_PROFILE)
        if (yakl_masterproc()) std::cout << "Using YAKL Timers via GPTL\n";
        GPTLinitialize();
      #endif
      bool use_pool = true;
      // #ifndef YAKL_SEPARATE_MEMORY_SPACE
      //   use_pool = false;
      // #endif

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

        std::function<void *( size_t )> alloc;
        std::function<void ( void * )>  dealloc;
        yakl::set_alloc_free(alloc , dealloc);
        pool.init(alloc,dealloc);

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
        yaklAllocDeviceFunc = [=] (size_t bytes , char const *label) -> void * { return alloc(bytes); };
        yaklFreeDeviceFunc  = [=] (void *ptr , char const *label)              { dealloc(ptr); };
      }

      yaklAllocHostFunc = [] (size_t bytes , char const *label) -> void * { return malloc(bytes); };
      yaklFreeHostFunc  = [] (void *ptr , char const *label) { free(ptr); };

      #ifdef YAKL_ARCH_CUDA
        cudaMalloc(&functorBuffer,functorBufSize);
      #endif

      #ifdef YAKL_ARCH_SYCL
        if (yakl_masterproc()) std::cout << "Running on "
                                         << sycl_default_stream().get_device().get_info<sycl::info::device::name>()
                                         << "\n";
        functorBuffer = sycl::malloc_device(functorBufSize, sycl_default_stream());
        fence();
      #endif

      #if defined(YAKL_ARCH_CUDA)
        int id;
        cudaGetDevice(&id);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props,id);
        if (yakl_masterproc()) std::cout << props.name << std::endl;
      #endif

      #if defined(YAKL_ARCH_HIP)
        int id;
        hipGetDevice(&id);
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props,id);
        if (yakl_masterproc()) std::cout << props.name << std::endl;
      #endif

      #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
        if (yakl_masterproc()) std::cout << "INFORM: Automatically inserting fence() after every parallel_for" << std::endl;
      #endif
    } else {
      std::cerr << "WARNING: Calling yakl::initialize() when YAKL is already initialized. ";
    }

    yakl_mtx.unlock();
  } //
