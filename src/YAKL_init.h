
#pragma once
// Included by YAKL.h

namespace yakl {

  inline void init( InitConfig config = InitConfig() ) {
    if constexpr (kokkos_debug) {
      if (!Kokkos::is_initialized()) Kokkos::abort("ERROR: yakl::init called before Kokkos::initialize");
    }
    // If YAKL is already initialized, then don't do anything
    if ( ! get_yakl_instance().is_initialized() ) {
      bool   pool_enabled     = true;
      size_t pool_bytes       = 4294967296;
      size_t pool_block_bytes = 4096;
      if (config.get_pool_size_mb()==0) {
        // Check if gator is disabled
        char * env = std::getenv("GATOR_DISABLE");
        if ( env != nullptr ) {
          std::string resp(env);
          if (resp == "yes" || resp == "YES" || resp == "1" || resp == "true" || resp == "TRUE" || resp == "T") {
            pool_enabled = false;
          }
        }
        // Check for GATOR_INITIAL_MB environment variable
        env = std::getenv("GATOR_INITIAL_MB");
        if ( env != nullptr ) {
          char * end = nullptr;
          errno = 0;
          long long size_mb = std::strtoll(env,&end,10);
          if (errno == 0 && end != env && *end == '\0' && size_mb > 0 &&
              static_cast<unsigned long long>(size_mb) <= std::numeric_limits<size_t>::max()/(1024*1024)) {
            pool_bytes = static_cast<size_t>(size_mb)*1024*1024;
          } else {
            if (yakl_mainproc()) std::cout << "WARNING: Invalid GATOR_INITIAL_MB. Defaulting to 1GB\n";
          }
        }
        // Check for GATOR_BLOCK_BYTES environment variable
        env = std::getenv("GATOR_BLOCK_BYTES");
        if ( env != nullptr ) {
          char * end = nullptr;
          errno = 0;
          long long block_bytes = std::strtoll(env,&end,10);
          if (errno == 0 && end != env && *end == '\0' && block_bytes > 0 &&
              static_cast<unsigned long long>(block_bytes) <= std::numeric_limits<size_t>::max() &&
              block_bytes%(2*sizeof(size_t)) == 0) {
            pool_block_bytes = static_cast<size_t>(block_bytes);
          } else {
            if (yakl_mainproc()) std::cout << "WARNING: Invalid GATOR_BLOCK_BYTES. Defaulting to 2*sizeof(size_t)\n";
            if (yakl_mainproc()) std::cout << "         GATOR_BLOCK_BYTES must be > 0 and a multiple of 2*sizeof(size_t)\n";
          }
        }
      } else {
        pool_enabled     = config.get_pool_enabled();
        if constexpr (kokkos_debug) {
          if (config.get_pool_size_mb() > std::numeric_limits<size_t>::max()/(1024*1024)) {
            Kokkos::abort("ERROR: configured pool size overflows size_t");
          }
        }
        pool_bytes       = config.get_pool_size_mb()*1024*1024;
        pool_block_bytes = config.get_pool_block_bytes();
        if (pool_block_bytes == 0 || pool_block_bytes%(2*sizeof(size_t)) != 0) pool_block_bytes = 4096;
      }

      get_yakl_instance().yakl_is_initialized = true;
      get_yakl_instance().pool_enabled = pool_enabled;

      if (get_yakl_instance().use_pool()) {
        auto alloc   = [] (size_t bytes) -> void * { return Kokkos::kokkos_malloc( "YAKL Pool allocation" , bytes ); };
        auto dealloc = [] (void *ptr) { Kokkos::kokkos_free( ptr ); };
        auto zero    = [] (void *ptr, size_t bytes) {};
        std::string error_message_out_of_memory = "You have run out of pool memory. Please use a larger pool size\n";
        get_yakl_instance().pool = LinearAllocator(pool_bytes,pool_block_bytes,alloc,dealloc,zero,"Gator",
                                                   error_message_out_of_memory);
        if (yakl_mainproc()) std::cout << "Using memory pool. Size: " << (float) pool_bytes/1024./1024./1024.
                                       << "GB." << std::endl;
      }
      if constexpr (yakl_auto_fence) {
        if (yakl_mainproc()) std::cout << "INFORM: Automatically inserting fence() after every yakl parallel_for"
                                       << std::endl;
      }
    } else {
      if constexpr (kokkos_debug) {
        std::cerr << "WARNING: Calling yakl::initialize() when YAKL is already initialized. ";
      }
    }
  } // init

} // namespace yakl
