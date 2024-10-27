/**
 * @file
 * An object of this class can optionally be passed to yakl::init() to configure the initialization
 */

#pragma once

#include <unistd.h>

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
    bool pool_enabled;
    /** @private */
    size_t pool_size_mb;
    /** @private */
    size_t pool_block_bytes;

  public:
    /** @brief Creating an InitConfig() controls the memory pool parameters, timer function overrides, and device
      *        allocation and deallocation overrides. **IMPORTANT**: Creating an InitConfig object pings environment
      *        variables, making it quite expensive to create. Please do not create a lot of these. */
    InitConfig() {
      pool_enabled     = true;
      pool_size_mb     = 4*1024;
      pool_block_bytes = 4096;

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
        long int size_mb = atol(env);
        if (size_mb != 0) {
          pool_size_mb = size_mb;
        } else {
          if (yakl_mainproc()) std::cout << "WARNING: Invalid GATOR_INITIAL_MB. Defaulting to 1GB\n";
        }
      }

      // Check for GATOR_BLOCK_BYTES environment variable
      env = std::getenv("GATOR_BLOCK_BYTES");
      if ( env != nullptr ) {
        long int block_bytes = atol(env);
        if (block_bytes != 0 && block_bytes%(2*sizeof(size_t)) == 0) {
          pool_block_bytes = block_bytes;
        } else {
          if (yakl_mainproc()) std::cout << "WARNING: Invalid GATOR_BLOCK_BYTES. Defaulting to 16*sizeof(size_t)\n";
          if (yakl_mainproc()) std::cout << "         GATOR_BLOCK_BYTES must be > 0 and a multiple of 2*sizeof(size_t)\n";
        }
      }
    }
    /** @brief Tell YAKL whether to enable the pool or not */
    InitConfig set_pool_enabled    ( bool enabled      ) { this->pool_enabled     = enabled    ; return *this; }
    /** @brief Tell YAKL how big the pool should be in MB */
    InitConfig set_pool_size_mb    ( size_t size_mb    ) { this->pool_size_mb     = size_mb    ; return *this; }
    /** @brief Tell YAKL how big each additional pool should be in MB */
    InitConfig set_pool_block_bytes( size_t block_bytes) { this->pool_block_bytes = block_bytes; return *this; }
    /** @brief Determine whether this config object will enable the device memory pool */
    bool   get_pool_enabled    () const { return pool_enabled    ; }
    /** @brief Determine how many MB this config will request the pool to use for the device memory pool */
    size_t get_pool_size_mb    () const { return pool_size_mb    ; }
    /** @brief Determine how many bytes this config will request the pool to use for block size */
    size_t get_pool_block_bytes() const { return pool_block_bytes; }
  };

}
