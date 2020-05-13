
#pragma once



struct SA_node {
  size_t start;
  size_t length;
};



class StackishAllocator {
protected:
  unsigned static constexpr SA_MAX = 1024;      // Maximum number of allocations allowed
  void *pool;                                   // Raw pool pointer
  size_t nBlocks;                               // Number of blocks in the pool
  unsigned blockSize;                           // Size of each block in bytes
  unsigned blockInc;                            // Pointer increment for a block if pointer is size_t *
  std::list<SA_node> allocs;                    // List of allocations
  std::function<void *( size_t )> mymalloc;     // allocation function
  std::function<void( void * )> myfree;         // free function
  std::function<void( void *, size_t )> myzero; // zero function


public:

  StackishAllocator() {
    this->blockSize = 0;
    this->nBlocks   = 0;
    this->pool      = nullptr;
  }


  StackishAllocator( size_t                                bytes ,
                     std::function<void *( size_t )>       mymalloc  = [] (size_t bytes) -> void * { return ::malloc(bytes); } ,
                     std::function<void( void * )>         myfree    = [] (void *ptr) { free(ptr); } ,
                     std::function<void( void *, size_t )> myzero    = [] (void *ptr, size_t bytes) {} ,
                     unsigned                              blockSize = 128*sizeof(size_t) ) {
    if (blockSize%sizeof(size_t) != 0) { yakl_throw("Error: blockSize must be a multiple of sizeof(size_t)"); }
    this->blockSize = blockSize;
    this->blockInc  = blockSize / sizeof(size_t);
    this->nBlocks   = (bytes-1) / blockSize + 1;
    this->mymalloc  = mymalloc;
    this->myfree    = myfree  ;
    this->myzero    = myzero  ;
    this->pool      = mymalloc(nBlocks*blockSize);
    this->zero( pool , nBlocks*blockSize );
  }


  ~StackishAlloctor() {
    myfree( this->pool );
  }


  void * allocate(size_t bytes) {
    if (bytes == 0) { return nullptr; }
    size_t blocksReq = (bytes-1)/blockSize + 1; // Number of blocks needed for this allocation
    if (allocs.empty()) {
      if (nBlocks >= blocksReq) {
        allocs.push_back( { (size_t) 0 , blocksReq } );
        return pool;
      } else {
        return nullptr;
      }
    } else {
      // If there's room at the end of the pool, then place the allocation there
      if ( nBlocks - (allocs.back().start + allocs.back().length) >= blocksReq ) {
        size_t newStart = allocs.back().start + allocs.back().length;
        allocs.push_back( { newStart , blocksReq } );
        return getPtr(newStart);
      } else {
        // If we start looking in between allocations, this could incur a large performance penalty
        return nullptr;
      }
    }
  };


  void deallocate(void *ptr) {
    // Iterate backwards from the end to search for the pointer
    // Efficiency requires stack-like accesses
    for (auto it = allocs.rbegin() ; it != allocs.rend() ; it++) {
      if ( ptr == getPtr((*it).start) ) { allocs.erase(it); return; }
    }
    yakl_throw("Error: Trying to deallocate an invalid pointer");
  };


  void * getPtr( size_t blockIndex ) {
    return (void *) ( ( (size_t *) pool ) + blockIndex*blockInc );
  }

};


