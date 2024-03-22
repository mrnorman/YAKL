
#pragma once

__YAKL_NAMESPACE_WRAPPER_BEGIN__
namespace yakl {

  /** @private */
  class YAKL_Internal {
    private:
      YAKL_Internal() {
        yakl_is_initialized = false;  // Determine if YAKL has been initialized
        timer_init_func = [] () {
          yakl_throw("ERROR: attempting to call the yakl::timer_init(); before calling yakl::init()");
        };
        timer_finalize_func = [] () {
          yakl_throw("ERROR: attempting to call the yakl::timer_finalize(); before calling yakl::init()");
        };
        timer_start_func = [] (char const *label) {
          yakl_throw("ERROR: attempting to call the yakl::timer_start(); before calling yakl::init()");
        };
        timer_stop_func = [] (char const * label) {
          yakl_throw("ERROR: attempting to call the yakl::timer_stop(); before calling yakl::init()");
        };
        alloc_device_func = [] ( size_t bytes , char const *label ) -> void* {
          yakl_throw("ERROR: attempting memory alloc before calling yakl::init()");
          return nullptr;
        };
        free_device_func  = [] ( void *ptr    , char const *label )          {
          yakl_throw("ERROR: attempting memory free before calling yakl::init()");
        };
        device_allocators_are_default  = false;
        pool_enabled                   = false;
      }
      ~YAKL_Internal() = default;

    public:
      YAKL_Internal(const YAKL_Internal&) = delete;
      YAKL_Internal& operator = (const YAKL_Internal&) = delete;

      Gator pool;                // Pool allocator. Constructor and destructor do not depend on ordering
      Toney timer;               // Constructor and destructor do not depend on ordering
      std::mutex yakl_mtx;       // Mutex for YAKL reference counting, allocation, and deallocation in threaded regions
      std::mutex yakl_final_mtx; // Mutex for YAKL reference counting, allocation, and deallocation in threaded regions
      bool yakl_is_initialized;  // Determine if YAKL has been initialized
      #ifdef YAKL_ARCH_HIP
        bool rocfft_is_initialized;
      #endif
      std::function<void ()> timer_init_func;              // Function to init timers
      std::function<void ()> timer_finalize_func;          // Function to finalize timers
      std::function<void (char const *)> timer_start_func; // Function to start a single timer
      std::function<void (char const *)> timer_stop_func;  // Function to stop a single timer
      std::function<void *( size_t , char const *)> alloc_device_func; // Function to alloc on device
      std::function<void ( void * , char const *)>  free_device_func;  // Funciton to free on device
      bool device_allocators_are_default;  // Are the allocators & deallocators default, or have they been changed?
      bool pool_enabled;                   // Is the pool allocator being used?
      std::vector< std::function<void ()> > finalize_callbacks;

      static YAKL_Internal & get_instance() {
        static YAKL_Internal instance;
        return instance;
      }
  };


  /** @private */
  inline YAKL_Internal & get_yakl_instance() { return YAKL_Internal::get_instance(); }
}
__YAKL_NAMESPACE_WRAPPER_END__

