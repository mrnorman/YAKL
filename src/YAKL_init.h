/**
 * @file
 * YAKL initialization routine
 */

#pragma once
// Included by YAKL.h

namespace yakl {

  /**
   * @brief Determine if the YAKL runtime has been initialized. I.e., yakl::init() has been called without a
   *        corresponding call to yakl::finalize().
   */
  inline bool isInitialized() { return get_yakl_instance().yakl_is_initialized; }


  /**
   * @brief Initialize the YAKL runtime.
   * 
   * 1. Determine if the pool allocator is to be used & pool allocator parameters.
   * 2. Initialize the pool if used.
   * 3. Inform the user with device information.
   * @param config This yakl::InitConfig object allows the user to override YAKL's default allocator, deallocator
   *               and timer calls from the start of the runtime.
   */
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
        size_t initialSize = config.get_pool_size_mb()*1024*1024;
        size_t blockSize   = config.get_pool_block_bytes();
        // Set the allocation and deallocation functions
        auto alloc   = [] (size_t bytes) -> void * {
          #ifdef YAKL_MANAGED_MEMORY
            return Kokkos::kokkos_malloc<Kokkos::SharedSpace>( "Pool allocation" , bytes );
          #else
            return Kokkos::kokkos_malloc( "Pool allocation" , bytes );
          #endif
        };
        auto dealloc = [] (void *ptr) { Kokkos::kokkos_free( ptr ); };
        auto zero    = [] (void *ptr, size_t bytes) {};
        std::string error_message_out_of_memory = "You have run out of pool memory. Please use a larger pool size\n";
        get_yakl_instance().pool = LinearAllocator(initialSize,blockSize,alloc,dealloc,zero,"Gator",
                                                   error_message_out_of_memory);
        if (yakl_mainproc()) std::cout << "Using memory pool. Size: " << (float) initialSize/1024./1024./1024.
                                       << "GB." << std::endl;
      }
      #if defined(YAKL_AUTO_FENCE)
        if (yakl_mainproc()) std::cout << "INFORM: Automatically inserting fence() after every yakl parallel_for"
                                       << std::endl;
      #endif
    } else {
      #ifdef KOKKOS_ENABLE_DEBUG
        std::cerr << "WARNING: Calling yakl::initialize() when YAKL is already initialized. ";
      #endif
    }

    yakl_mtx_unlock();
  } //
}


