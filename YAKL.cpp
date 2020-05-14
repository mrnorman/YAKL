
#include "YAKL.h"

namespace yakl {

  void *functorBuffer;

  StackyAllocator pool;

  // YAKL allocator and deallocator
  std::function<void *( size_t )> yaklAllocDeviceFunc = [] ( size_t bytes ) -> void* {std::cout << "ERROR: attempting memory alloc before calling yakl::init()\n"; exit(-1);};
  std::function<void ( void * )>  yaklFreeDeviceFunc  = [] ( void *ptr    )          {std::cout << "ERROR: attempting memory free before calling yakl::init()\n"; exit(-1);};

  // YAKL allocator and deallocator
  std::function<void *( size_t )> yaklAllocHostFunc = [] ( size_t bytes ) -> void* {std::cout << "ERROR: attempting memory alloc before calling yakl::init()\n"; exit(-1);};
  std::function<void ( void * )>  yaklFreeHostFunc  = [] ( void *ptr    )          {std::cout << "ERROR: attempting memory free before calling yakl::init()\n"; exit(-1);};


  void *yaklAllocDevice( size_t bytes ) {
    return yaklAllocDeviceFunc(bytes);
  }


  void yaklFreeDevice( void *ptr ) {
    yaklFreeDeviceFunc(ptr);
  }


  void *yaklAllocHost( size_t bytes ) {
    return yaklAllocHostFunc(bytes);
  }


  void yaklFreeHost( void *ptr ) {
    yaklFreeHostFunc(ptr);
  }

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


