
#pragma once

#include "YAKL_alloc_free.h"

  // Initialize the YAKL framework
  inline void init() {
    #ifdef YAKL_PROFILE
      std::cout << "Using YAKL Timers via GPTL\n";
      GPTLinitialize();
    #endif
    bool use_pool = true;

    #ifdef YAKL_ARCH_SYCL
      auto asyncHandler = [&](sycl::exception_list eL) {
        for (auto& e : eL) {
          try {
            std::rethrow_exception(e);
          } catch (sycl::exception& e) {
            std::cout << e.what() << std::endl;
            std::cout << "fail" << std::endl;
            std::terminate();
          }
        }
      };

      sycl::default_selector device_selector;
      sycl_default_stream = sycl::queue(device_selector, asyncHandler,
                                        sycl::property_list{sycl::property::queue::in_order{}});
      std::cout << "Running on "
                << sycl_default_stream.get_device().get_info<sycl::info::device::name>()
                << "\n";
    #endif

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

    #if defined(YAKL_ARCH_CUDA)
      cudaMalloc(&functorBuffer,functorBufSize);
    #endif

    #if defined(YAKL_ARCH_HIP)
      int id;
      hipGetDevice(&id);
      hipDeviceProp_t props;
      hipGetDeviceProperties(&props,id);
      std::cout << props.name << std::endl;
    #endif

    #if defined(YAKL_AUTO_FENCE) || defined(YAKL_DEBUG)
      std::cout << "INFORM: Automatically inserting fence() after every parallel_for" << std::endl;
    #endif

  } //
