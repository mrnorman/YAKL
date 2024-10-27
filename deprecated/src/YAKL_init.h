/**
 * @file
 * YAKL initialization routine
 */

#pragma once
// Included by YAKL.h

__YAKL_NAMESPACE_WRAPPER_BEGIN__
namespace yakl {

  /**
   * @brief Determine if the YAKL runtime has been initialized. I.e., yakl::init() has been called without a
   *        corresponding call to yakl::finalize().
   */
  inline bool isInitialized() { return get_yakl_instance().yakl_is_initialized; }


  /**
   * @brief Initialize the YAKL runtime.
   * 
   * 1. Determin if the pool allocator is to be used & pool allocator parameters.
   * 2. Initialize the pool if used.
   * 3. Set the YAKL allocators and deallocators to default. 
   * 4. Initialize YAKL's timer calls to defaults.
   * 5. Inspect the optional yakl::InitConfig parameter to override default allocator, deallocator,
   *    and timer calls if requested.
   * 6. Inform the user with device information. THREAD SAFE!
   * @param config This yakl::InitConfig object allows the user to override YAKL's default allocator, deallocator
   *               and timer calls from the start of the runtime.
   */
  // Set global std::functions for alloc and free
  inline void init( InitConfig config = InitConfig() ) {
    yakl_mtx_lock();

    // If YAKL is already initialized, then don't do anything
    if ( ! isInitialized() ) {
      #if defined(YAKL_PROFILE)
        if (yakl_mainproc()) std::cout << "Using YAKL Timers\n";
      #endif

      get_yakl_instance().yakl_is_initialized = true;

      get_yakl_instance().pool_enabled = config.get_pool_enabled();

      // Initialize the memory pool and default allocators
      if (use_pool()) {
        // Set pool defaults if environment variables are not set
        size_t initialSize = config.get_pool_initial_mb()*1024*1024;
        size_t growSize    = config.get_pool_grow_mb()*1024*1024;
        size_t blockSize   = config.get_pool_block_bytes();

        // Set the allocation and deallocation functions
        std::function<void *( size_t )> alloc;
        std::function<void ( void * )>  dealloc;
        set_device_alloc_free(alloc , dealloc);
        auto zero = [] (void *ptr, size_t bytes) {};
        std::string pool_name = "Gator: YAKL's primary memory pool";
        std::string error_message_out_of_memory = "Please see https://github.com/mrnorman/YAKL/wiki/Pool-Allocator for more information\n";
        std::string error_message_cannot_grow = error_message_out_of_memory;
        get_yakl_instance().pool.init(alloc,dealloc,zero,initialSize,growSize,blockSize,pool_name,
                                      error_message_out_of_memory,error_message_cannot_grow);
        if (yakl_mainproc()) std::cout << "Using memory pool. Initial size: " << (float) initialSize/1024./1024./1024.
                                       << "GB ;  Grow size: " << (float) growSize/1024./1024./1024. << "GB." << std::endl;
      }

      set_yakl_allocators_to_default();

      // Initialize the default timers
      get_yakl_instance().timer_init_func = [] () {};
      get_yakl_instance().timer_finalize_func = [] () {
        #if defined(YAKL_PROFILE)
          get_yakl_instance().timer.print_all_threads();
        #endif
      };
      get_yakl_instance().timer_start_func = [] (char const *label) {
        #if defined(YAKL_PROFILE)
          fence();  get_yakl_instance().timer.start( label );
        #endif
      };
      get_yakl_instance().timer_stop_func = [] (char const * label) {
        #if defined(YAKL_PROFILE)
          fence();  get_yakl_instance().timer.stop( label );
        #endif
      };

      get_yakl_instance().device_allocators_are_default = true;

      // If the user specified overrides in the InitConfig, apply them here
      if (config.get_device_allocator  ()) {
        get_yakl_instance().alloc_device_func = config.get_device_allocator  ();
        get_yakl_instance().device_allocators_are_default = false;
      }
      if (config.get_device_deallocator()) {
        get_yakl_instance().free_device_func  = config.get_device_deallocator();
        get_yakl_instance().device_allocators_are_default = false;
      }
      if (config.get_timer_init        ()) get_yakl_instance().timer_init_func      = config.get_timer_init    ();
      if (config.get_timer_finalize    ()) get_yakl_instance().timer_finalize_func  = config.get_timer_finalize();
      if (config.get_timer_start       ()) get_yakl_instance().timer_start_func     = config.get_timer_start   ();
      if (config.get_timer_stop        ()) get_yakl_instance().timer_stop_func      = config.get_timer_stop    ();

      #ifdef YAKL_ARCH_SYCL
        if (yakl_mainproc()) std::cout << "Running on "
                                       << sycl_default_stream().get_device().get_info<sycl::info::device::name>()
                                       << "\n";
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
        get_yakl_instance().rocfft_is_initialized = false;
        if (yakl_mainproc()) std::cout << props.name << std::endl;
      #endif

      #if defined(YAKL_AUTO_FENCE)
        if (yakl_mainproc()) std::cout << "INFORM: Automatically inserting fence() after every parallel_for"
                                       << std::endl;
      #endif
      #ifdef YAKL_VERBOSE_FILE
        int rank = 0;
        #ifdef HAVE_MPI
          int is_initialized;
          MPI_Initialized(&is_initialized);
          if (is_initialized) { MPI_Comm_rank(MPI_COMM_WORLD, &rank); }
        #endif
        std::ofstream myfile;
        std::string fname = std::string("yakl_verbose_output_task_") + std::to_string(rank) + std::string(".log");
        myfile.open(fname , std::ofstream::out);
        myfile.close();
      #endif
    } else {
      std::cerr << "WARNING: Calling yakl::initialize() when YAKL is already initialized. ";
    }

    yakl_mtx_unlock();
  } //
}
__YAKL_NAMESPACE_WRAPPER_END__


