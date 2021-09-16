
#include "YAKL.h"

namespace yakl {

  #ifdef YAKL_ARCH_SYCL
    sycl::queue sycl_default_stream;
  #endif

  std::mutex yakl_mtx;

  Gator pool;

  void * functorBuffer;

  bool yakl_is_initialized = false;

  // YAKL allocator and deallocator
  std::function<void *( size_t , char const *)> yaklAllocDeviceFunc = [] ( size_t bytes , char const *label ) -> void* {
    yakl_throw("ERROR: attempting memory alloc before calling yakl::init()");
    return nullptr;
  };
  std::function<void ( void * , char const *)>  yaklFreeDeviceFunc  = [] ( void *ptr    , char const *label )          {
    yakl_throw("ERROR: attempting memory free before calling yakl::init()");
  };

  // YAKL allocator and deallocator
  std::function<void *( size_t , char const *)> yaklAllocHostFunc = [] ( size_t bytes , char const *label ) -> void* {
    yakl_throw("ERROR: attempting memory alloc before calling yakl::init()");
    return nullptr;
  };
  std::function<void ( void * , char const *)>  yaklFreeHostFunc  = [] ( void *ptr    , char const *label )          {
    yakl_throw("ERROR: attempting memory free before calling yakl::init()");
  };


  #if defined(YAKL_ARCH_HIP) || defined(YAKL_ARCH_SYCL)
    // YAKL_INLINE void *yaklAllocDevice( size_t bytes , char const *label ) { return yaklAllocDeviceFunc(bytes,label); }
    // YAKL_INLINE void yaklFreeDevice( void *ptr , char const *label ) { yaklFreeDeviceFunc(ptr,label); }
    // YAKL_INLINE void *yaklAllocHost( size_t bytes , char const *label ) { return yaklAllocHostFunc(bytes,label); }
    // YAKL_INLINE void yaklFreeHost( void *ptr , char const *label ) { yaklFreeHostFunc(ptr,label); }
    // YAKL_INLINE void yakl_mtx_lock()   { yakl_mtx.lock(); }
    // YAKL_INLINE void yakl_mtx_unlock() { yakl_mtx.unlock(); }
  #else
    void *yaklAllocDevice( size_t bytes , char const *label ) { return yaklAllocDeviceFunc(bytes,label); }
    void yaklFreeDevice( void *ptr , char const *label ) { yaklFreeDeviceFunc(ptr,label); }
    void *yaklAllocHost( size_t bytes , char const *label ) { return yaklAllocHostFunc(bytes,label); }
    void yaklFreeHost( void *ptr , char const *label ) { yaklFreeHostFunc(ptr,label); }
    void yakl_mtx_lock()   { yakl_mtx.lock(); }
    void yakl_mtx_unlock() { yakl_mtx.unlock(); }
  #endif

}


extern "C" void gatorInit() {
  yakl::init();
}


extern "C" void gatorFinalize() {
  yakl::finalize();
}


// Fortran binding
extern "C" void* gatorAllocate( size_t bytes ) {
  return yakl::yaklAllocDevice( bytes , "");
}


// Fortran binding
extern "C" void gatorDeallocate( void *ptr ) {
  yakl::yaklFreeDevice( ptr , "");
}


