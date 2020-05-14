
#pragma once

#include "StackyAllocator.h"


class Gator {
protected:
  std::vector<StackyAllocator> pools;             // The pools managed by this class
  std::function<void *( size_t )>       mymalloc; // allocation function
  std::function<void( void * )>         myfree;   // free function
  std::function<void( void *, size_t )> myzero;   // zero function
  size_t growSize;
  size_t blockSize;


public:

  Gator() {
    std::function<void *( size_t )> alloc;
    std::function<void ( void * )>  dealloc;

    #if   defined(__USE_CUDA__)
      #if defined (__MANAGED__)
        alloc   = [] ( size_t bytes ) -> void* {
          void *ptr;
          cudaMallocManaged(&ptr,bytes);
          cudaMemPrefetchAsync(ptr,bytes,0);
          #ifdef _OPENMP45
            omp_target_associate_ptr(ptr,ptr,bytes,0,0);
          #endif
          #ifdef _OPENACC
            acc_map_data(ptr,ptr,bytes);
          #endif
          return ptr;
        };
        dealloc = [] ( void *ptr    ) {
          cudaFree(ptr);
        };
      #else
        alloc   = [] ( size_t bytes ) -> void* {
          void *ptr;
          cudaMalloc(&ptr,bytes);
          return ptr;
        };
        dealloc = [] ( void *ptr    ) {
          cudaFree(ptr);
        };
      #endif
    #elif defined(__USE_HIP__)
      #if defined (__MANAGED__)
        alloc   = [] ( size_t bytes ) -> void* { void *ptr; hipMallocHost(&ptr,bytes); return ptr; };
        dealloc = [] ( void *ptr    )          { hipFree(ptr); };
      #else
        alloc   = [] ( size_t bytes ) -> void* { void *ptr; hipMalloc(&ptr,bytes); return ptr; };
        dealloc = [] ( void *ptr    )          { hipFree(ptr); };
      #endif
    #else
      alloc   = ::malloc;
      dealloc = ::free;
    #endif
    init(alloc,dealloc);
  }


  Gator( std::function<void *( size_t )>       mymalloc ,
         std::function<void( void * )>         myfree   ,
         std::function<void( void *, size_t )> myzero    = [] (void *ptr, size_t bytes) {} ) {
    init(mymalloc,myfree,myzero);
  }


  Gator            (      Gator && );
  Gator &operator= (      Gator && );
  Gator            (const Gator &  ) = delete;
  Gator &operator= (const Gator &  ) = delete;


  ~Gator() { finalize(); }


  static constexpr const char *classname() { return "Gator"; }


  void init( std::function<void *( size_t )>       mymalloc  = [] (size_t bytes) -> void * { return ::malloc(bytes); } ,
             std::function<void( void * )>         myfree    = [] (void *ptr) { ::free(ptr); } ,
             std::function<void( void *, size_t )> myzero    = [] (void *ptr, size_t bytes) {} ) {
    this->mymalloc = mymalloc;
    this->myfree   = myfree  ;
    this->myzero   = myzero  ;

    // Default to 1GB initial size and grow size
    size_t initialSize = 1024*1024*1024;
    this->growSize     = initialSize;
    this->blockSize    = sizeof(size_t)*128;

    // Check for GATOR_INITIAL_MB environment variable
    char * env = std::getenv("GATOR_INITIAL_MB");
    if ( env != nullptr ) {
      long int initial_mb = atol(env);
      if (initial_mb != 0) {
        initialSize = initial_mb;
        this->growSize = initialSize;
      } else {
        std::cout << "WARNING: Invalid GATOR_INITIAL_MB. Defaulting to 1GB\n";
      }
    }

    // Check for GATOR_GROW_MB environment variable
    env = std::getenv("GATOR_GROW_MB");
    if ( env != nullptr ) {
      long int grow_mb = atol(env);
      if (grow_mb != 0) {
        this->growSize = grow_mb;
      } else {
        std::cout << "WARNING: Invalid GATOR_GROW_MB. Defaulting to 1GB\n";
      }
    }

    // Check for GATOR_BLOCK_BYTES environment variable
    env = std::getenv("GATOR_BLOCK_BYTES");
    if ( env != nullptr ) {
      long int block_bytes = atol(env);
      if (block_bytes != 0 && block_bytes%sizeof(size_t) == 0) {
        this->blockSize = block_bytes;
      } else {
        std::cout << "WARNING: Invalid GATOR_BLOCK_BYTES. Defaulting to 128*sizeof(size_t)\n";
        std::cout << "         GATOR_BLOCK_BYTES must be > 0 and a multiple of sizeof(size_t)\n";
      }
    }

    pools.push_back(StackyAllocator(initialSize , mymalloc , myfree , blockSize , myzero));
  }


  void finalize() {
    pools = std::vector<StackyAllocator>();
  }


  void * allocate(size_t bytes, std::string label="") {
    // Loop through the pools and see if there's room. If so, allocate in one of them
    for (int i=0; i < pools.size(); i++) {
      if (pools[i].iGotRoom(bytes)) { return allocate(bytes,label); }
    }
    // If we're here, ther isn't enough room in the existing pools. We need to create a new one
    pools.push_back(StackyAllocator(growSize , mymalloc , myfree , blockSize , myzero));
    return pools.back().allocate(bytes,label);
  };


  void free(void *ptr) {
    // Iterate backwards. It's assumed accesses are stack-like
    for (int i=pools.size()-1; i >=0; i--) {
      if (pools[i].thisIsMyPointer(ptr)) { pools[i].free(ptr); }
    }
    throw("Error: Trying to free an invalid pointer");
  };


  size_t highWaterMark() const {
    size_t highWater = 0;
    for (int i=0; i <= pools.size() ; i++) { highWater += pools[i].highWaterMark(); }
    return highWater;
  }


  size_t poolSize( ) const {
    size_t sz = 0;
    for (int i=0; i <= pools.size() ; i++) { sz += pools[i].poolSize(); }
    return sz;
  }


  size_t numAllocs( ) const {
    size_t allocs = 0;
    for (int i=0; i <= pools.size() ; i++) { allocs += pools[i].numAllocs(); }
    return allocs;
  }

};


