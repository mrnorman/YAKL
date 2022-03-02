
#pragma once
// Included by YAKL.h

namespace yakl {
  // Initialize the YAKL framework
  // Initialize timers, set global std::functions for alloc and free, allocate functorBuffer
  inline void init() {
    yakl_mtx.lock();

    // If YAKL is already initialized, then don't do anything
    if ( ! isInitialized() ) {
      #if defined(YAKL_PROFILE) || defined(YAKL_AUTO_PROFILE)
        if (yakl_masterproc()) std::cout << "Using YAKL Timers\n";
        timer_init();
      #endif
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

        // Set the allocation and deallocation functions
        std::function<void *( size_t )> alloc;
        std::function<void ( void * )>  dealloc;
        yakl::set_alloc_free(alloc , dealloc);
        pool.init(alloc,dealloc);

        // Assign the global std::functions for device allocation and free
        // Perform all allocs and frees through the pool
        yaklAllocDevice = [] (size_t bytes , char const *label) -> void * {
          return pool.allocate( bytes , label );
        };
        yaklFreeDevice  = [] (void *ptr , char const *label)              {
          pool.free( ptr , label );
        };

      } else {

        // Set the allocation and deallocation functions
        std::function<void *( size_t)> alloc;
        std::function<void ( void *)>  dealloc;
        set_alloc_free(alloc , dealloc);

        // Assign the global std::functions for device allocation and free
        yaklAllocDevice = [=] (size_t bytes , char const *label) -> void * { return alloc(bytes); };
        yaklFreeDevice  = [=] (void *ptr , char const *label)              { dealloc(ptr); };

      }

      // Assign the global std::functions for host allocation and free
      yaklAllocHost = [] (size_t bytes , char const *label) -> void * { return malloc(bytes); };
      yaklFreeHost  = [] (void *ptr , char const *label) { free(ptr); };

      // Allocate functorBuffer
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

      // Print the device name being run on
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
        if (yakl_masterproc()) std::cout << "INFORM: Automatically inserting fence() after every parallel_for"
                                         << std::endl;
      #endif
    } else {
      std::cerr << "WARNING: Calling yakl::initialize() when YAKL is already initialized. ";
    }

    yakl_mtx.unlock();
  } //
}


