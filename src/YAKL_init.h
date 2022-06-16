
#pragma once
// Included by YAKL.h

namespace yakl {

  extern bool yakl_is_initialized;

  inline bool isInitialized() { return yakl_is_initialized; }

  // Initialize the YAKL framework
  // Set global std::functions for alloc and free, allocate functorBuffer
  inline void init( InitConfig config = InitConfig() ) {
    yakl_mtx.lock();

    // If YAKL is already initialized, then don't do anything
    if ( ! isInitialized() ) {
      #if defined(YAKL_PROFILE)
        if (yakl_mainproc()) std::cout << "Using YAKL Timers\n";
      #endif

      yakl_is_initialized = true;

      // Initialize the memory pool and default allocators
      if (use_pool()) {
        // Set the allocation and deallocation functions
        std::function<void *( size_t )> alloc;
        std::function<void ( void * )>  dealloc;
        set_device_alloc_free(alloc , dealloc);
        pool.init(alloc,dealloc);
      }

      set_yakl_allocators_to_default();

      // Initialize the default timers
      timer_init = [] () {};
      timer_finalize = [] () {
        #if defined(YAKL_PROFILE)
          if (yakl_mainproc()) { timer.print_all_threads(); }
        #endif
      };
      timer_start = [] (char const *label) {
        #if defined(YAKL_PROFILE)
          fence();  timer.start( label );
        #endif
      };
      timer_stop = [] (char const * label) {
        #if defined(YAKL_PROFILE)
          fence();  timer.stop( label );
        #endif
      };

      // If the user specified overrides in the InitConfig, apply them here
      if (config.get_host_allocator    ()) yaklAllocHost   = config.get_host_allocator    ();
      if (config.get_device_allocator  ()) yaklAllocDevice = config.get_device_allocator  ();
      if (config.get_host_deallocator  ()) yaklFreeHost    = config.get_host_deallocator  ();
      if (config.get_device_deallocator()) yaklFreeDevice  = config.get_device_deallocator();
      if (config.get_timer_init        ()) timer_init      = config.get_timer_init        ();
      if (config.get_timer_finalize    ()) timer_finalize  = config.get_timer_finalize    ();
      if (config.get_timer_start       ()) timer_start     = config.get_timer_start       ();
      if (config.get_timer_stop        ()) timer_stop      = config.get_timer_stop        ();

      // Allocate functorBuffer
      #ifdef YAKL_ARCH_CUDA
        cudaMalloc(&functorBuffer,functorBufSize);
        fence();
      #endif
      #ifdef YAKL_ARCH_SYCL
        if (yakl_mainproc()) std::cout << "Running on "
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
        if (yakl_mainproc()) std::cout << props.name << std::endl;
      #endif

      #if defined(YAKL_ARCH_HIP)
        int id;
        hipGetDevice(&id);
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props,id);
        if (yakl_mainproc()) std::cout << props.name << std::endl;
      #endif

      #if defined(YAKL_AUTO_FENCE)
        if (yakl_mainproc()) std::cout << "INFORM: Automatically inserting fence() after every parallel_for"
                                       << std::endl;
      #endif
    } else {
      std::cerr << "WARNING: Calling yakl::initialize() when YAKL is already initialized. ";
    }

    yakl_mtx.unlock();
  } //
}


