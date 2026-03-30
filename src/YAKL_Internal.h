
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
      bool            yakl_is_initialized;  // Determine if YAKL has been initialized
      bool            pool_enabled;         // Is the pool allocator being used?

      bool use_pool      () const { return pool_enabled; }
      bool get_pool      () const { return pool_enabled; }
      bool is_initialized() const { return yakl_is_initialized; }

      static YAKL_Internal & get_instance() {
        static YAKL_Internal instance;
        return instance;
      }
  };


  /** @private */
  inline YAKL_Internal & get_yakl_instance() { return YAKL_Internal::get_instance(); }
}

