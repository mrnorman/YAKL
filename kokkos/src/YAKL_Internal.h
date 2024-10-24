
#pragma once

namespace yakl {

  /** @private */
  class YAKL_Internal {
    private:
      YAKL_Internal() {
        yakl_is_initialized = false;  // Determine if YAKL has been initialized
        pool_enabled        = false;
      }
      ~YAKL_Internal() = default;

    public:
      YAKL_Internal(const YAKL_Internal&) = delete;
      YAKL_Internal& operator = (const YAKL_Internal&) = delete;

      LinearAllocator pool;                 // Pool allocator. Constructor and destructor do not depend on ordering
      Toney           timer;                // Constructor and destructor do not depend on ordering
      std::mutex      yakl_mtx;             // Mutex for YAKL reference counting, allocation, and deallocation in threaded regions
      std::mutex      yakl_final_mtx;       // Mutex for YAKL reference counting, allocation, and deallocation in threaded regions
      bool            yakl_is_initialized;  // Determine if YAKL has been initialized
      bool            pool_enabled;         // Is the pool allocator being used?

      static YAKL_Internal & get_instance() {
        static YAKL_Internal instance;
        return instance;
      }
  };


  /** @private */
  inline YAKL_Internal & get_yakl_instance() { return YAKL_Internal::get_instance(); }
}

