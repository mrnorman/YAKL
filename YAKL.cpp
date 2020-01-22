
#include "YAKL.h"

namespace yakl {

  void *functorBuffer;

  BuddyAllocator pool;

  // YAKL allocator and deallocator
  std::function<void *( size_t )> yaklAllocDevice;
  std::function<void ( void * )>  yaklFreeDevice;

  // YAKL allocator and deallocator
  std::function<void *( size_t )> yaklAllocHost;
  std::function<void ( void * )>  yaklFreeHost;

}


// Fortran binding
extern "C" void* gatorAllocate( size_t bytes ) {
  return yakl::pool.allocate( bytes );
}


// Fortran binding
extern "C" void gatorDeallocate( void *ptr ) {
  yakl::pool.free( ptr );
}


