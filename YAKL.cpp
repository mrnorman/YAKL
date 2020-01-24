
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


extern "C" void gatorInit() {
  yakl::init();
}


extern "C" void gatorInitPool( size_t bytes ) {
  yakl::init( bytes );
}


extern "C" void gatorFinalize( size_t bytes ) {
  yakl::finalize();
}


// Fortran binding
extern "C" void* gatorAllocate( size_t bytes ) {
  return yakl::yaklAllocDevice( bytes );
}


// Fortran binding
extern "C" void gatorDeallocate( void *ptr ) {
  yakl::yaklFreeDevice( ptr );
}


