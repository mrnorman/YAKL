/**
 * @file
 * An object of this class can optionally be passed to yakl::init() to configure the initialization
 */

#pragma once

namespace yakl {
  
  /** @brief An object of this class can optionally be passed to yakl::init() to configure the initialization.
    *        **IMPORTANT**: Creating an InitConfig object pings environment
    *        variables, making it quite expensive to create. Please do not create a lot of these.
    * @details This allows the user to override timer, allocation, and deallocation routines.
    * 
    * All `set_` functions return the InitConfig object they were called on. Therefore, the user can code, e.g.,
    * `yakl::init(yakl::InitConfig().set_device_allocator(myalloc).set_device_deallocator(myfree));` */
  class InitConfig {
  protected:
    /** @private */
    std::function<void *( size_t , char const *)> alloc_device_func;
    /** @private */
    std::function<void ( void * , char const *)>  free_device_func;  
    /** @private */
    std::function<void ()>                        timer_init;
    /** @private */
    std::function<void ()>                        timer_finalize;
    /** @private */
    std::function<void (char const *)>            timer_start;
    /** @private */
    std::function<void (char const *)>            timer_stop;
    /** @private */
    bool pool_enabled;
    /** @private */
    size_t pool_initial_mb;
    /** @private */
    size_t pool_grow_mb;
    /** @private */
    size_t pool_block_bytes;
  
  public:
    /** @brief Creating an InitConfig() controls the memory pool parameters, timer function overrides, and device
      *        allocation and deallocation overrides. **IMPORTANT**: Creating an InitConfig object pings environment
      *        variables, making it quite expensive to create. Please do not create a lot of these. */
    InitConfig() {
      pool_enabled     = true;
      pool_initial_mb  = 1024;
      pool_grow_mb     = 1024;
      pool_block_bytes = 16*sizeof(size_t);

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
        long int initial_mb = atol(env);
        if (initial_mb != 0) {
          pool_initial_mb = initial_mb;
          pool_grow_mb = pool_initial_mb;
        } else {
          if (yakl::yakl_mainproc()) std::cout << "WARNING: Invalid GATOR_INITIAL_MB. Defaulting to 1GB\n";
        }
      }
      // Check for GATOR_GROW_MB environment variable
      env = std::getenv("GATOR_GROW_MB");
      if ( env != nullptr ) {
        long int grow_mb = atol(env);
        if (grow_mb != 0) {
          pool_grow_mb = grow_mb;
        } else {
          if (yakl::yakl_mainproc()) std::cout << "WARNING: Invalid GATOR_GROW_MB. Defaulting to 1GB\n";
        }
      }

      // Check for GATOR_BLOCK_BYTES environment variable
      env = std::getenv("GATOR_BLOCK_BYTES");
      if ( env != nullptr ) {
        long int block_bytes = atol(env);
        if (block_bytes != 0 && block_bytes%(2*sizeof(size_t)) == 0) {
          pool_block_bytes = block_bytes;
        } else {
          if (yakl::yakl_mainproc()) std::cout << "WARNING: Invalid GATOR_BLOCK_BYTES. Defaulting to 16*sizeof(size_t)\n";
          if (yakl::yakl_mainproc()) std::cout << "         GATOR_BLOCK_BYTES must be > 0 and a multiple of 2*sizeof(size_t)\n";
        }
      }
    }
    /** @brief Pass the device allocator function you wish to use to override YAKL's default (NO LABEL) */
    InitConfig set_device_allocator( std::function<void *( size_t )> func ) {
      alloc_device_func = [=] (size_t bytes , char const *label) -> void * { return func(bytes); };
      return *this;
    }
    /** @brief Pass the device deallocator function you wish to use to override YAKL's default (NO LABEL) */
    InitConfig set_device_deallocator( std::function<void ( void * )> func ) {
      free_device_func  = [=] (void *ptr , char const *label) { func(ptr); };
      return *this;
    }
    /** @brief Pass the device allocator function you wish to use to override YAKL's default (LABEL) */
    InitConfig set_device_allocator  ( std::function<void *( size_t , char const *)> func ) { alloc_device_func = func; return *this; }
    /** @brief Pass the device deallocator function you wish to use to override YAKL's default (LABEL) */
    InitConfig set_device_deallocator( std::function<void ( void * , char const *)>  func ) { free_device_func  = func; return *this; }
    /** @brief Pass the timer init function you wish to use to override YAKL's default */
    InitConfig set_timer_init        ( std::function<void (            )>            func ) { timer_init      = func; return *this; }
    /** @brief Pass the timer finalize function you wish to use to override YAKL's default */
    InitConfig set_timer_finalize    ( std::function<void (            )>            func ) { timer_finalize  = func; return *this; }
    /** @brief Pass the timer start function you wish to use to override YAKL's default */
    InitConfig set_timer_start       ( std::function<void (char const *)>            func ) { timer_start     = func; return *this; }
    /** @brief Pass the timer stop function you wish to use to override YAKL's default */
    InitConfig set_timer_stop        ( std::function<void (char const *)>            func ) { timer_stop      = func; return *this; }
    /** @brief Tell YAKL whether to enable the pool or not */
    InitConfig set_pool_enabled    ( bool enabled      ) { this->pool_enabled     = enabled    ; return *this; }
    /** @brief Tell YAKL how big the initial pool should be in MB */
    InitConfig set_pool_initial_mb ( size_t initial_mb ) { this->pool_initial_mb  = initial_mb ; return *this; }
    /** @brief Tell YAKL how big each additional pool should be in MB */
    InitConfig set_pool_grow_mb    ( size_t grow_mb    ) { this->pool_grow_mb     = grow_mb    ; return *this; }
    /** @brief Tell YAKL how big each additional pool should be in MB */
    InitConfig set_pool_block_bytes( size_t block_bytes) { this->pool_block_bytes = block_bytes; return *this; }
    /** @brief Get the device allocator function. Returns an empty std::function if the user has not set one */
    std::function<void *( size_t , char const *)> get_device_allocator  () const { return alloc_device_func; }
    /** @brief Get the device deallocator function. Returns an empty std::function if the user has not set one */
    std::function<void ( void * , char const *)>  get_device_deallocator() const { return free_device_func ; }
    /** @brief Get the timer init function. Returns an empty std::function if the user has not set one */
    std::function<void ()>                        get_timer_init        () const { return timer_init     ; }
    /** @brief Get the timer finalize function. Returns an empty std::function if the user has not set one */
    std::function<void ()>                        get_timer_finalize    () const { return timer_finalize ; }
    /** @brief Get the timer start function. Returns an empty std::function if the user has not set one */
    std::function<void (char const *)>            get_timer_start       () const { return timer_start    ; }
    /** @brief Get the timer stop function. Returns an empty std::function if the user has not set one */
    std::function<void (char const *)>            get_timer_stop        () const { return timer_stop     ; }
    /** @brief Determine whether this config object will enable the device memory pool */
    bool   get_pool_enabled    () const { return pool_enabled    ; }
    /** @brief Determine how many MB this config will request the pool to use for the initial device memory pool */
    size_t get_pool_initial_mb () const { return pool_initial_mb ; }
    /** @brief Determine how many MB this config will request the pool to use for additional pools */
    size_t get_pool_grow_mb    () const { return pool_grow_mb    ; }
    /** @brief Determine how many bytes this config will request the pool to use for block size */
    size_t get_pool_block_bytes() const { return pool_block_bytes; }
  };

}


