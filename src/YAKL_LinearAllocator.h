
#pragma once

struct AllocNode {
  size_t start;       // Offset of this allocation in "blocks"
  size_t length;      // Length of this allocation in "blocks"
  char const * label;  // Label for this allocation
  AllocNode() {
    this->start  = 0;
    this->length = 0;
    this->label  = "";
  }
  AllocNode( size_t start , size_t length , char const * label ) {
    this->start  = start;
    this->length = length;
    this->label  = label;
  }
};

class LinearAllocator {
public:

  void                                  *pool;     // Raw pool pointer
  size_t                                nBlocks;   // Number of blocks in the pool
  unsigned                              blockSize; // Size of each block in bytes
  unsigned                              blockInc;  // Number of size_t in each block
  std::vector<AllocNode>                allocs;    // List of allocations
  std::function<void *( size_t )>       mymalloc;  // allocation function
  std::function<void( void * )>         myfree;    // free function
  std::function<void( void *, size_t )> myzero;    // zero function


  LinearAllocator() { nullify(); }


  LinearAllocator( size_t                                bytes ,
                   unsigned                              blockSize = sizeof(size_t) ,
                   std::function<void *( size_t )>       mymalloc  = [] (size_t bytes) -> void * { return ::malloc(bytes); } ,
                   std::function<void( void * )>         myfree    = [] (void *ptr) { ::free(ptr); } ,
                   std::function<void( void *, size_t )> myzero    = [] (void *ptr, size_t bytes) {} ) {
    nullify();

    if (yakl::yakl_masterproc()) std::cout << "Create Pool\n";
    if (blockSize%sizeof(size_t) != 0) { die("Error: blockSize must be a multiple of sizeof(size_t)"); }
    this->blockSize = blockSize;
    this->blockInc  = blockSize / sizeof(size_t);
    this->nBlocks   = (bytes-1) / blockSize + 1;
    this->mymalloc  = mymalloc;
    this->myfree    = myfree  ;
    this->myzero    = myzero  ;
    this->pool      = mymalloc( poolSize() );
    this->allocs    = std::vector<AllocNode>();
    this->allocs.reserve(128);
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
  }


  LinearAllocator( LinearAllocator && rhs) {
    this->pool      = rhs.pool     ;
    this->nBlocks   = rhs.nBlocks  ;
    this->blockSize = rhs.blockSize;
    this->blockInc  = rhs.blockInc ;
    this->allocs    = rhs.allocs   ;
    this->mymalloc  = rhs.mymalloc ;
    this->myfree    = rhs.myfree   ;
    this->myzero    = rhs.myzero   ;
    rhs.nullify();
  }


  LinearAllocator &operator =( LinearAllocator && rhs) {
    if (this == &rhs) { return *this; }
    this->finalize();
    this->pool      = rhs.pool     ;
    this->nBlocks   = rhs.nBlocks  ;
    this->blockSize = rhs.blockSize;
    this->blockInc  = rhs.blockInc ;
    this->allocs    = rhs.allocs   ;
    this->mymalloc  = rhs.mymalloc ;
    this->myfree    = rhs.myfree   ;
    this->myzero    = rhs.myzero   ;
    rhs.nullify();
    return *this;
  }


  LinearAllocator( LinearAllocator const &rhs ) = delete;


  LinearAllocator &operator=( LinearAllocator const &rhs ) = delete;


  ~LinearAllocator() {
    if (pool != nullptr) {
      if (yakl::yakl_masterproc()) std::cout << "Destroy Pool\n";
    }
    finalize();
  }


  void nullify() {
    this->pool      = nullptr;
    this->nBlocks   = 0;
    this->blockSize = 0;
    this->blockInc  = 0;
    this->allocs    = std::vector<AllocNode>();
    this->mymalloc  = [] (size_t bytes) -> void * { return ::malloc(bytes); };
    this->myfree    = [] (void *ptr) { ::free(ptr); };
    this->myzero    = [] (void *ptr, size_t bytes) {};
  }


  void finalize() {
    if (allocs.size() != 0) {
      #if defined(YAKL_DEBUG) || defined(MEMORY_DEBUG)
        std::cerr << "WARNING: Not all allocations were deallocated before destroying this pool.\n" << std::endl;
        printAllocsLeft();
        std::cerr << "This probably won't end well, but carry on.\n" << std::endl;
      #endif
    }
    if (this->pool != nullptr) { myfree( this->pool ); }
    nullify();
  }


  void printAllocsLeft() {
    if (allocs.size() != 0) {
      std::cerr << "The following allocations have not been deallocated:" << std::endl;
      for (int i=0; i < allocs.size(); i++) {
        std::cerr << "*** Label: " << allocs[i].label
                  << "  ;  size: " << allocs[i].length*blockSize
                  << " bytes  ;  offset: " << allocs[i].start*blockSize
                  << " bytes  ;  ptr: " << getPtr(allocs[i].start) << std::endl;
      }
    }
  }


  static constexpr const char *classname() { return "LinearAllocator"; }


  void * allocate(size_t bytes, char const * label="") {
    #ifdef MEMORY_DEBUG
      if (yakl::yakl_masterproc()) std::cout << "MEMORY DEBUG: LinearAllocator attempting to allocate "
                                             << label << " with " << bytes << " bytes\n";
    #endif
    if (bytes == 0) {
      return nullptr;
    }
    size_t blocksReq = (bytes-1)/blockSize + 1; // Number of blocks needed for this allocation
    if (allocs.empty()) {
      if (nBlocks >= blocksReq) {
        allocs.push_back( AllocNode( (size_t) 0 , blocksReq , label ) );
        return pool;
      }
    } else {
      // Look for room before the first allocation
      if ( allocs.front().start >= blocksReq ) {
        allocs.insert( allocs.begin() , AllocNode( 0 , blocksReq , label ) );
        return getPtr(allocs[0].start);
      }

      // Loop through the allocations and look for free space between this and the next
      for (int i=0; i < allocs.size()-1; i++) {
        if ( allocs[i+1].start - (allocs[i].start + allocs[i].length) >= blocksReq ) {
          allocs.insert( allocs.begin()+i+1 , AllocNode( allocs[i].start+allocs[i].length , blocksReq , label ) );
          return getPtr(allocs[i+1].start);
        }
      }

      // Look for room after the last allocation
      if ( nBlocks - (allocs.back().start + allocs.back().length) >= blocksReq ) {
        allocs.push_back( AllocNode( allocs.back().start + allocs.back().length , blocksReq , label ) );
        return getPtr(allocs.back().start);
      }
    }

    return nullptr;
  };


  void free(void *ptr, char const * label = "") {
    #ifdef MEMORY_DEBUG
      if (yakl::yakl_masterproc()) std::cout << "MEMORY DEBUG: LinearAllocator attempting to free "
                                             << label << " with the pointer: " << ptr << "\n";
    #endif
    for (int i=allocs.size()-1; i >= 0; i--) {
      if (ptr == getPtr(allocs[i].start)) {
        allocs.erase(allocs.begin()+i);
        return;
      }
    }
    die("Error: LinearAllocator is trying to free an invalid pointer");
  };


  bool iGotRoom( size_t bytes ) const {
    size_t blocksReq = (bytes-1)/blockSize + 1; // Number of blocks needed for this allocation

    if (allocs.empty()) {
      if (nBlocks >= blocksReq) { return true; }
    } else {
      // Look for room before the first allocation
      if ( allocs.front().start >= blocksReq ) { return true; }

      // Loop through the allocations and look for free space between this and the next
      for (int i=0; i < allocs.size()-1; i++) {
        if ( allocs[i+1].start - (allocs[i].start + allocs[i].length) >= blocksReq ) { return true; }
      }

      // Look for room after the last allocation
      if ( nBlocks - (allocs.back().start + allocs.back().length) >= blocksReq ) { return true; }
    }

    return false;
  }


  bool thisIsMyPointer(void *ptr) const {
    long long offset = ( (size_t *) ptr - (size_t *) pool ) / blockInc;
    return (offset >= 0 && offset <= nBlocks-1);
  }


  bool initialized() const { return pool != nullptr; }


  size_t poolSize() const { return nBlocks*blockSize; }


  size_t numAllocs() const { return allocs.size(); }


  // Transform a block index into a memory pointer
  void * getPtr( size_t blockIndex ) const {
    return (void *) ( ( (size_t *) pool ) + blockIndex*blockInc );
  }


  void die(std::string str="") {
    std::cerr << str << std::endl;
    throw str;
  }


};


