
#include "YAKL.h"

namespace yakl {

  void *functorBuffer;

  BuddyAllocator pool;

  // YAKL allocator and deallocator
  std::function<void *( size_t )> yaklAllocDevice = [] ( size_t bytes ) -> void* {std::cout << "ERROR: attempting memory alloc before calling yakl::init()\n"; exit(-1);};
  std::function<void ( void * )>  yaklFreeDevice  = [] ( void *ptr    )          {std::cout << "ERROR: attempting memory free before calling yakl::init()\n"; exit(-1);};

  // YAKL allocator and deallocator
  std::function<void *( size_t )> yaklAllocHost = [] ( size_t bytes ) -> void* {std::cout << "ERROR: attempting memory alloc before calling yakl::init()\n"; exit(-1);};
  std::function<void ( void * )>  yaklFreeHost  = [] ( void *ptr    )          {std::cout << "ERROR: attempting memory free before calling yakl::init()\n"; exit(-1);};

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


