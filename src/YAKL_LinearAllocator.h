
#pragma once
// This is the YAKL memory pool implementation. It's slow, but it honestly doesn't need to be fast
// because memory allocations will generally be overlapped with GPU kernel execution.

namespace yakl {


  // This class encapsulates a single "pool"
  class LinearAllocator {
  public:

    // Describes a single allocation entry
    struct AllocNode {
      size_t      start;  // Offset of this allocation in "blocks"
      size_t      length; // Length of this allocation in "blocks"
      std::string label;  // Label for this allocation
      AllocNode() {
        this->start  = 0;
        this->length = 0;
        this->label  = "";
      }
      AllocNode( size_t start , size_t length , std::string label ) {
        this->start  = start;
        this->length = length;
        this->label  = std::move(label);
      }
    };

    std::string                           pool_name;
    void                                  *pool;     // Raw pool pointer
    size_t                                nBlocks;   // Number of blocks in the pool
    size_t                                blockSize; // Size of each block in bytes
    std::vector<AllocNode>                allocs;    // List of allocations
    std::function<void *( size_t )>       mymalloc;  // allocation function
    std::function<void( void * )>         myfree;    // free function
    std::function<void( void *, size_t )> myzero;    // zero function


    LinearAllocator() { nullify(); }


    LinearAllocator( size_t                                bytes ,
                     size_t                                blockSize = 16*sizeof(size_t) ,
                     std::function<void * ( size_t )>      mymalloc  = [] (size_t bytes) -> void * { return ::malloc(bytes); } ,
                     std::function<void( void * )>         myfree    = [] (void *ptr) { ::free(ptr); } ,
                     std::function<void( void *, size_t )> myzero    = [] (void *ptr, size_t bytes) {} ,
                     std::string                           pool_name = "Gator" ,
                     std::string                           error_message_out_of_memory = "" ) {
      if (bytes == 0) Kokkos::abort("ERROR: Attempting to create a memory pool with zero bytes");
      nullify();

      if (blockSize%(2*sizeof(size_t)) != 0) {
        std::cerr << "ERROR: Pool labeled \"" << pool_name << "\" -> LinearAllocator:" << std::endl;
        Kokkos::abort("Error: LinearAllocator blockSize must be a multiple of 2*sizeof(size_t)");
      }
      this->blockSize = blockSize;
      this->nBlocks   = (bytes-1) / blockSize + 1;
      this->mymalloc  = mymalloc;
      this->myfree    = myfree  ;
      this->myzero    = myzero  ;
      this->pool      = mymalloc( poolSize() );
      this->allocs    = std::vector<AllocNode>();
      this->allocs.reserve(128);  // Make sure there is initial room for 128 entries
      this->pool_name = pool_name;
      if (pool == nullptr) {
        std::cerr << "ERROR: Pool labeled \"" << pool_name << "\" -> LinearAllocator:" << std::endl;
        std::cerr << "Could not create pool of size " << bytes << " bytes (" << bytes/1024./1024./1024. << " GB)."
                  << "\nYou have run out of memory." << std::endl;
        Kokkos::abort( error_message_out_of_memory.c_str() );
      }
      this->myzero( pool , poolSize() );
    }


    // Allow the pool to be moved, but not copied
    LinearAllocator( LinearAllocator && rhs) {
      this->pool      = rhs.pool     ;
      this->nBlocks   = rhs.nBlocks  ;
      this->blockSize = rhs.blockSize;
      this->allocs    = std::move(rhs.allocs   );
      this->mymalloc  = std::move(rhs.mymalloc );
      this->myfree    = std::move(rhs.myfree   );
      this->myzero    = std::move(rhs.myzero   );
      this->pool_name = std::move(rhs.pool_name);
      rhs.nullify();
    }


    LinearAllocator &operator =( LinearAllocator && rhs) {
      if (this == &rhs) { return *this; }
      this->finalize();
      this->pool      = rhs.pool     ;
      this->nBlocks   = rhs.nBlocks  ;
      this->blockSize = rhs.blockSize;
      this->allocs    = std::move(rhs.allocs   );
      this->mymalloc  = std::move(rhs.mymalloc );
      this->myfree    = std::move(rhs.myfree   );
      this->myzero    = std::move(rhs.myzero   );
      this->pool_name = std::move(rhs.pool_name);
      rhs.nullify();
      return *this;
    }


    LinearAllocator( LinearAllocator const &rhs ) = delete;


    LinearAllocator &operator=( LinearAllocator const &rhs ) = delete;


    ~LinearAllocator() {
      finalize();
    }


    void nullify() {
      this->pool      = nullptr;
      this->nBlocks   = 0;
      this->blockSize = 0;
      this->allocs    = std::vector<AllocNode>();
      this->mymalloc  = [] (size_t bytes) -> void * { return ::malloc(bytes); };
      this->myfree    = [] (void *ptr) { ::free(ptr); };
      this->myzero    = [] (void *ptr, size_t bytes) {};
    }


    void finalize() {
      if (allocs.size() != 0) {
        if constexpr (kokkos_debug) {
          std::cerr << "WARNING: Pool labeled \"" << pool_name << "\" -> LinearAllocator:" << std::endl;
          std::cerr << "WARNING: Not all allocations were deallocated before destroying this pool.\n" << std::endl;
          printAllocsLeft();
          std::cerr << "This probably won't end well, but carry on.\n" << std::endl;
        }
      }
      if (this->pool != nullptr) { myfree( this->pool ); }
      nullify();
    }


    // Mostly for debug purposes. Print all existing allocations
    void printAllocsLeft() const {
      if (allocs.size() != 0) {
        std::cerr << "The following allocations have not been deallocated:" << std::endl;
        for (size_t i=0; i < allocs.size(); i++) {
          std::cerr << "*** Label: "         << allocs[i].label
                    << "  ;  size: "         << allocs[i].length*blockSize
                    << " bytes  ;  offset: " << allocs[i].start*blockSize
                    << " bytes  ;  ptr: "    << getPtr(allocs[i].start) << std::endl;
        }
      }
    }


    // Allocate the requested number of bytes if there is room for it.
    // If there isn't room or bytes == 0, the simulation is aborted.
    // Otherwise, the correct pointer is returned
    void * allocate(size_t bytes, std::string label="") {
      if (bytes == 0) return nullptr;
      size_t blocksReq = (bytes-1)/blockSize + 1; // Number of blocks needed for this allocation
      // If there are no allocations, then place this allocaiton at the beginning.
      if (allocs.empty()) {
        if (nBlocks >= blocksReq) {
          allocs.push_back( AllocNode( (size_t) 0 , blocksReq , std::move(label) ) );
          return pool;
        }
      } else {
        // Look for room before the first allocation
        if ( allocs.front().start >= blocksReq ) {
          allocs.insert( allocs.begin() , AllocNode( 0 , blocksReq , std::move(label) ) );
          return getPtr(allocs[0].start);
        }
        // Loop through the allocations and look for free space between this and the next
        for (size_t i=0; i+1 < allocs.size(); i++) {
          if ( allocs[i+1].start - (allocs[i].start + allocs[i].length) >= blocksReq ) {
            allocs.insert( allocs.begin()+i+1 , AllocNode( allocs[i].start+allocs[i].length , blocksReq , std::move(label) ) );
            return getPtr(allocs[i+1].start);
          }
        }
        // Look for room after the last allocation
        if ( nBlocks - (allocs.back().start + allocs.back().length) >= blocksReq ) {
          allocs.push_back( AllocNode( allocs.back().start + allocs.back().length , blocksReq , std::move(label) ) );
          return getPtr(allocs.back().start);
        }
      }

      Kokkos::abort( "The pool has run out of memory. Please initialize a larger pool." );
      return nullptr;
    };


    // Free the requested pointer
    // Returns the number of bytes in the allocation being freed
    size_t free(void * ptr, std::string label = "") {
      for (size_t i=0; i < allocs.size(); i++) {
        if (ptr == getPtr(allocs[i].start)) {
          size_t bytes = allocs[i].length*blockSize;
          allocs.erase(allocs.begin()+i);
          return bytes;
        }
      }
      std::cerr << "ERROR: Pool labeled \"" << pool_name << "\" -> LinearAllocator: ["
                                            << label << "]: " << std::endl;
      std::cerr << "Trying to free an invalid pointer.\n";
      Kokkos::abort("This means you have either already freed the pointer, or its address has been corrupted somehow.");
      return 0;
    };


    // Determine if there is room for an allocation of the requested number of bytes
    bool iGotRoom( size_t bytes ) const {
      if (bytes == 0) return true;
      size_t blocksReq = (bytes-1)/blockSize + 1; // Number of blocks needed for this allocation
      if (allocs.empty()) {
        if (nBlocks >= blocksReq) { return true; }
      } else {
        // Look for room before the first allocation
        if ( allocs.front().start >= blocksReq ) { return true; }
        // Loop through the allocations and look for free space between this and the next
        for (size_t i=0; i+1 < allocs.size(); i++) {
          if ( allocs[i+1].start - (allocs[i].start + allocs[i].length) >= blocksReq ) { return true; }
        }
        // Look for room after the last allocation
        if ( nBlocks - (allocs.back().start + allocs.back().length) >= blocksReq ) { return true; }
      }
      return false;
    }


    // Determine if the requested pointer belongs to this pool
    bool thisIsMyPointer(void * ptr_in) const {
      std::uintptr_t ptr   = reinterpret_cast<uintptr_t>(ptr_in);
      std::uintptr_t start = reinterpret_cast<uintptr_t>(pool  );
      std::uintptr_t end   = start + poolSize();
      return ptr >= start && ptr < end;
    }


    bool initialized() const { return pool != nullptr; }


    size_t poolSize() const {
      return nBlocks*blockSize;
    }


    size_t numAllocs() const { return allocs.size(); }


    // Transform a block index into a memory pointer
    void * getPtr( size_t blockIndex ) const {
      return static_cast<char*>(pool) + blockIndex * blockSize;
    }


  };

}


