
#pragma once

#include <iostream>
#include <iomanip>
#include <list>
#include <functional>



struct SA_node {
  size_t start;       // Offset of this allocation in "blocks"
  size_t length;      // Length of this allocation in "blocks"
  char const * label;  // Label for this allocation
};



class StackyAllocator {
protected:
  void *pool;                                   // Raw pool pointer
  size_t nBlocks;                               // Number of blocks in the pool
  unsigned blockSize;                           // Size of each block in bytes
  unsigned blockInc;                            // Pointer increment for a block if pointer is size_t *
  std::list<SA_node> allocs;                    // List of allocations
  std::function<void *( size_t )> mymalloc;     // allocation function
  std::function<void( void * )> myfree;         // free function
  std::function<void( void *, size_t )> myzero; // zero function
  int *refCount; // Pointer shared by multiple copies of this StackyAllocator to keep track of allcation / free
  size_t highWater;                             // Memory high-water mark in blocks

  // Transform a block index into a memory pointer
  void * getPtr( size_t blockIndex ) const {
    return (void *) ( ( (size_t *) pool ) + blockIndex*blockInc );
  }

  void die(std::string str="") {
    std::cerr << str << std::endl;
    throw str;
  }


public:

  StackyAllocator() {
    this->highWater = 0;
    this->blockSize = 0;
    this->nBlocks   = 0;
    this->pool      = nullptr;
    this->refCount  = nullptr;
  }


  StackyAllocator( size_t                                bytes ,
                   std::function<void *( size_t )>       mymalloc  = [] (size_t bytes) -> void * { return ::malloc(bytes); } ,
                   std::function<void( void * )>         myfree    = [] (void *ptr) { ::free(ptr); } ,
                   unsigned                              blockSize = 128*sizeof(size_t) ,
                   std::function<void( void *, size_t )> myzero    = [] (void *ptr, size_t bytes) {} ) {
    if (blockSize%sizeof(size_t) != 0) { die("Error: blockSize must be a multiple of sizeof(size_t)"); }
    this->highWater = 0;
    this->blockSize = blockSize;
    this->blockInc  = blockSize / sizeof(size_t);
    this->nBlocks   = (bytes-1) / blockSize + 1;
    this->mymalloc  = mymalloc;
    this->myfree    = myfree  ;
    this->myzero    = myzero  ;
    this->pool      = mymalloc( poolSize() );
    if (pool == nullptr) {
      std::cerr << "ERROR: Could not create pool of size " << bytes << "\n" <<
                   "You have run out of memory. If GATOR_INITIAL_MB and GATOR_GROW_MB are too small " <<
                   "and your variables are too large, you're using each of the pools inefficiently. " <<
                   "To fix this, you need to increase GATOR_INITIAL_MB to a larger value rather than " <<
                   "relying on a lot of pools. Otherwise, you'll just have to reduce the problem size " <<
                   "per node." << std::endl;
      die();
    }
    this->myzero( pool , poolSize() );
    refCount = new int;
    *refCount = 1;
  }


  StackyAllocator( StackyAllocator && rhs) {
    this->highWater = rhs.highWater;
    this->pool      = rhs.pool     ;
    this->nBlocks   = rhs.nBlocks  ;
    this->blockSize = rhs.blockSize;
    this->blockInc  = rhs.blockInc ;
    this->allocs    = rhs.allocs   ;
    this->mymalloc  = rhs.mymalloc ;
    this->myfree    = rhs.myfree   ;
    this->myzero    = rhs.myzero   ;
    this->refCount  = rhs.refCount ;
    rhs.refCount = nullptr;
    rhs.pool     = nullptr;
    rhs.allocs   = std::list<SA_node>();
  }


  StackyAllocator &operator =( StackyAllocator && rhs) {
    if (this == &rhs) { return *this; }
    this->finalize();
    this->highWater = rhs.highWater;
    this->pool      = rhs.pool     ;
    this->nBlocks   = rhs.nBlocks  ;
    this->blockSize = rhs.blockSize;
    this->blockInc  = rhs.blockInc ;
    this->allocs    = rhs.allocs   ;
    this->mymalloc  = rhs.mymalloc ;
    this->myfree    = rhs.myfree   ;
    this->myzero    = rhs.myzero   ;
    this->refCount  = rhs.refCount ;
    rhs.refCount = nullptr;
    rhs.pool     = nullptr;
    rhs.allocs   = std::list<SA_node>();
    return *this;
  }


  StackyAllocator( StackyAllocator const &rhs ) {
    this->highWater = rhs.highWater;
    this->pool      = rhs.pool     ;
    this->nBlocks   = rhs.nBlocks  ;
    this->blockSize = rhs.blockSize;
    this->blockInc  = rhs.blockInc ;
    this->allocs    = rhs.allocs   ;
    this->mymalloc  = rhs.mymalloc ;
    this->myfree    = rhs.myfree   ;
    this->myzero    = rhs.myzero   ;
    this->refCount  = rhs.refCount ;
    (*refCount)++;
  }


  StackyAllocator &operator=( StackyAllocator const &rhs ) {
    if (this == &rhs) { return *this; }
    this->finalize();
    this->highWater = rhs.highWater;
    this->pool      = rhs.pool     ;
    this->nBlocks   = rhs.nBlocks  ;
    this->blockSize = rhs.blockSize;
    this->blockInc  = rhs.blockInc ;
    this->allocs    = rhs.allocs   ;
    this->mymalloc  = rhs.mymalloc ;
    this->myfree    = rhs.myfree   ;
    this->myzero    = rhs.myzero   ;
    this->refCount  = rhs.refCount ;
    (*refCount)++;
    return *this;
  }


  ~StackyAllocator() {
    finalize();
  }


  void finalize() {
    if (allocs.size() != 0) {
      std::cerr << "WARNING: Not all allocations were deallocated before destroying this pool.\n" <<
                   "The following allocations were not deallocated:" << std::endl;
      for (auto it = allocs.begin() ; it != allocs.end() ; it++) {
        std::cerr << "*** Label: " << it->label << "  ;  size: " << it->length*blockSize << " bytes  ;  offset: " << 
                     it->start*blockSize << " bytes  ;  ptr: " << getPtr(it->start) << std::endl;
      }
      std::cerr << "This probably won't end well, but carry on.\n" << std::endl;
    }
    if (refCount != nullptr) {
      (*refCount)--;

      if (*refCount == 0) {
        delete refCount;
        refCount = nullptr;
        allocs = std::list<SA_node>();
        if (this->pool != nullptr) { myfree( this->pool ); pool = nullptr; }
      }
    }
  }


  void checkAllocsLeft() {
    if (allocs.size() != 0) {
      std::cerr << "The following allocations were not deallocated:" << std::endl;
      for (auto it = allocs.begin() ; it != allocs.end() ; it++) {
        std::cerr << "*** Label: " << it->label << "  ;  size: " << it->length*blockSize << " bytes  ;  offset: " << 
                     it->start*blockSize << " bytes  ;  ptr: " << getPtr(it->start) << std::endl;
      }
    }
  }


  static constexpr const char *classname() { return "StackyAllocator"; }


  void * allocate(size_t bytes, char const * label="") {
    #ifdef MEMORY_DEBUG
      std::cout << "MEMORY DEBUG: StackyAllocator attempting to allocate " << label << " with " << bytes << " bytes\n";
    #endif
    if (bytes == 0) {
      return nullptr;
    }
    size_t blocksReq = (bytes-1)/blockSize + 1; // Number of blocks needed for this allocation
    if (allocs.empty()) {
      if (nBlocks >= blocksReq) {
        allocs.push_back( { (size_t) 0 , blocksReq , label } );
        if (allocs.back().start + allocs.back().length  > highWater) { highWater = allocs.back().start + allocs.back().length; }
        // std::cout << "Allocating: " << label << " with " << blocksReq*blockSize << " bytes at location " << pool << "\n";
        return pool;
      } else {
        return nullptr;
      }
    } else {
      // If there's room at the end of the pool, then place the allocation there
      if ( nBlocks - (allocs.back().start + allocs.back().length) >= blocksReq ) {
        size_t newStart = allocs.back().start + allocs.back().length;
        allocs.push_back( { newStart , blocksReq , label } );
        if (allocs.back().start + allocs.back().length  > highWater) { highWater = allocs.back().start + allocs.back().length; }
        // std::cout << "Allocating: " << label << " with " << blocksReq*blockSize << " bytes at location " << getPtr(newStart) << "\n";
        return getPtr(newStart);
      } else {
        // If we start looking in between allocations, this could incur a large performance penalty
        return nullptr;
      }
    }
  };


  void free(void *ptr, char const * label = "") {
    #ifdef MEMORY_DEBUG
      std::cout << "MEMORY DEBUG: StackyAllocator attempting to free " << label << " with the pointer: " << ptr << "\n";
    #endif
    // Iterate backwards from the end to search for the pointer
    // Efficiency requires stack-like accesses to avoid traversing the entire list
    for (auto it = allocs.rbegin() ; it != allocs.rend() ; it++) {
      if ( ptr == getPtr(it->start) ) {
        // std::cout << "Deallocating: " << it->label << "\n";
        allocs.erase(std::next(it).base());  // This syntax is really dumb. Just...why?
        return;
      }
    }
    die("Error: StackyAllocator is trying to free an invalid pointer");
  };


  size_t highWaterMark() const {
    return this->highWater*blockSize;
  }


  bool iGotRoom( size_t bytes ) const {
    size_t blocksReq = (bytes-1)/blockSize + 1; // Number of blocks needed for this allocation
    if (allocs.empty()) {
      if (nBlocks >= blocksReq) {
        return true;
      } else {
        return false;
      }
    } else {
      if ( nBlocks - (allocs.back().start + allocs.back().length) >= blocksReq ) {
        return true;
      } else {
        return false;
      }
    }
  }


  bool thisIsMyPointer(void *ptr) const {
    long long offset = ( (size_t *) ptr - (size_t *) pool ) / blockInc;
    return (offset >= 0 && offset <= nBlocks-1);
  }


  bool initialized() const {
    return pool != nullptr;
  }


  size_t poolSize( ) const {
    return nBlocks*blockSize;
  }


  size_t numAllocs( ) const {
    return allocs.size();
  }


  int use_count() const {
    if (refCount != nullptr) {
      return *refCount;
    } else {
      return 0;
    }
  }


};


