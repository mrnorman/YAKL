
#pragma once
// This is the YAKL memory pool implementation. It's slow, but it honestly doesn't need to be fast
// because memory allocations will generally be overlapped with GPU kernel execution.

namespace yakl {


  // This class owns one contiguous allocation and serves variable-size requests. Internally, each request is rounded up
  // to a whole number of fixed-size bookkeeping blocks.
  //
  // Invariants:
  //   * pool is aligned to requiredAlignment, and blockSize is a multiple of requiredAlignment.
  //     Therefore, every pointer returned at a block boundary satisfies Kokkos's alignment requirement.
  //   * allocs is maintained in ascending start-block order by inserting each new entry at its address position. Entries
  //     describe non-overlapping block ranges.
  //   * poolAllocation is the pointer returned by mymalloc and is the only pointer passed to myfree.
  //     pool may be advanced within that allocation to provide alignment.
  //   * Public operations lock mutex, so allocation metadata is safe for concurrent host callers.
  class LinearAllocator {
  public:

    size_t static constexpr requiredAlignment = Kokkos::Impl::MEMORY_ALIGNMENT;
    static_assert((requiredAlignment & (requiredAlignment-1)) == 0,"ERROR: LinearAllocator alignment must be a power of two");

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
    void                                  *poolAllocation; // Raw owning pointer returned by mymalloc
    void                                  *pool;           // Aligned start of the usable pool
    size_t                                nBlocks;         // Number of usable blocks
    size_t                                blockSize;       // Bytes per block
    std::vector<AllocNode>                allocs;          // Live allocations maintained in ascending start-block order
    std::function<void *( size_t )>       mymalloc;        // Backing allocation function
    std::function<void( void * )>         myfree;          // Backing deallocation function
    std::function<void( void *, size_t )> myzero;          // Optional backing-memory initialization function
    // Several locked public methods call other locked methods. A recursive mutex keeps those internal calls safe while
    // serializing concurrent host allocations, frees, queries, moves, and finalization.
    mutable std::recursive_mutex          mutex;


    LinearAllocator() { nullify(); }


    LinearAllocator( size_t                                bytes ,
                     size_t                                blockSize = requiredAlignment ,
                     std::function<void * ( size_t )>      mymalloc  = [] (size_t bytes) -> void * { return ::malloc(bytes); } ,
                     std::function<void( void * )>         myfree    = [] (void *ptr) { ::free(ptr); } ,
                     std::function<void( void *, size_t )> myzero    = [] (void *ptr, size_t bytes) {} ,
                     std::string                           pool_name = "Gator" ,
                     std::string                           error_message_out_of_memory = "" ) {
      if (bytes == 0) Kokkos::abort("ERROR: Attempting to create a memory pool with zero bytes");
      nullify();

      // A positive aligned block size makes the base of every suballocation Kokkos-aligned.
      if (blockSize == 0 || blockSize%requiredAlignment != 0) {
        std::cerr << "ERROR: Pool labeled \"" << pool_name << "\" -> LinearAllocator:" << std::endl;
        Kokkos::abort("ERROR: LinearAllocator blockSize must be a positive multiple of Kokkos memory alignment");
      }
      if (!mymalloc || !myfree || !myzero) Kokkos::abort("ERROR: LinearAllocator received an empty callback");
      // Rounding bytes up to a whole block must not overflow.
      if (bytes > std::numeric_limits<size_t>::max()-(blockSize-1)) {
        Kokkos::abort("ERROR: LinearAllocator pool-size rounding overflow");
      }
      this->blockSize = blockSize;
      this->nBlocks   = (bytes-1) / blockSize + 1;
      this->mymalloc  = mymalloc;
      this->myfree    = myfree  ;
      this->myzero    = myzero  ;
      // Reserve enough padding to align pool even when mymalloc provides weaker alignment. Keep poolAllocation unchanged
      // because only the original pointer may be passed to myfree.
      if (poolSize() > std::numeric_limits<size_t>::max()-(requiredAlignment-1)) {
        Kokkos::abort("ERROR: LinearAllocator aligned pool-size overflow");
      }
      this->poolAllocation = mymalloc( poolSize()+requiredAlignment-1 );
      if (poolAllocation == nullptr) {
        std::cerr << "ERROR: Pool labeled \"" << pool_name << "\" -> LinearAllocator:" << std::endl;
        std::cerr << "Could not create pool of size " << bytes << " bytes (" << bytes/1024./1024./1024. << " GB)."
                  << "\nYou have run out of memory." << std::endl;
        Kokkos::abort( error_message_out_of_memory.c_str() );
      }
      auto const rawAddress = reinterpret_cast<uintptr_t>(poolAllocation);
      if (rawAddress > std::numeric_limits<uintptr_t>::max()-(requiredAlignment-1)) {
        Kokkos::abort("ERROR: LinearAllocator pool-address alignment overflow");
      }
      auto const poolAddress = (rawAddress+requiredAlignment-1) & ~(requiredAlignment-1);
      this->pool      = reinterpret_cast<void *>(poolAddress);
      this->allocs    = std::vector<AllocNode>();
      this->allocs.reserve(128);  // Avoid reallocating metadata for typical allocation counts
      this->pool_name = pool_name;
      this->myzero( pool , poolSize() );
    }


    // Allow the pool to be moved, but not copied
    LinearAllocator( LinearAllocator && rhs) {
      std::lock_guard<std::recursive_mutex> lock(rhs.mutex);
      this->poolAllocation = rhs.poolAllocation;
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
      std::scoped_lock lock(this->mutex,rhs.mutex);
      this->finalize();
      this->poolAllocation = rhs.poolAllocation;
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
      std::lock_guard<std::recursive_mutex> lock(mutex);
      this->poolAllocation = nullptr;
      this->pool      = nullptr;
      this->nBlocks   = 0;
      this->blockSize = 0;
      this->allocs    = std::vector<AllocNode>();
      this->mymalloc  = [] (size_t bytes) -> void * { return ::malloc(bytes); };
      this->myfree    = [] (void *ptr) { ::free(ptr); };
      this->myzero    = [] (void *ptr, size_t bytes) {};
    }


    void finalize() {
      std::lock_guard<std::recursive_mutex> lock(mutex);
      if (allocs.size() != 0) {
        if constexpr (kokkos_debug) {
          std::cerr << "WARNING: Pool labeled \"" << pool_name << "\" -> LinearAllocator:" << std::endl;
          std::cerr << "WARNING: Not all allocations were deallocated before destroying this pool.\n" << std::endl;
          printAllocsLeft();
          std::cerr << "This probably won't end well, but carry on.\n" << std::endl;
        }
      }
      if (this->poolAllocation != nullptr) { myfree( this->poolAllocation ); }
      nullify();
    }


    // Mostly for debug purposes. Print all existing allocations
    void printAllocsLeft() const {
      std::lock_guard<std::recursive_mutex> lock(mutex);
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


    // Allocate the requested bytes with a first-fit search through the address-ordered allocation list.
    // Zero-byte allocations return nullptr. A nonzero request aborts if it cannot fit.
    void * allocate(size_t bytes, std::string label="") {
      std::lock_guard<std::recursive_mutex> lock(mutex);
      if (bytes == 0) return nullptr;
      if constexpr (kokkos_debug) {
        if (!initialized()) Kokkos::abort("ERROR: allocating from an uninitialized LinearAllocator");
      }
      if (bytes > std::numeric_limits<size_t>::max()-(blockSize-1)) {
        Kokkos::abort("ERROR: LinearAllocator allocation-size rounding overflow");
      }
      size_t blocksReq = (bytes-1)/blockSize + 1; // Number of whole blocks needed for this allocation
      // If there are no allocations, place this allocation at the beginning.
      if (allocs.empty()) {
        if (nBlocks >= blocksReq) {
          allocs.push_back( AllocNode( (size_t) 0 , blocksReq , std::move(label) ) );
          return pool;
        }
      } else {
        // Look for room before the first allocation.
        if ( allocs.front().start >= blocksReq ) {
          allocs.insert( allocs.begin() , AllocNode( 0 , blocksReq , std::move(label) ) );
          return getPtr(allocs[0].start);
        }
        // Look for a large enough interior gap.
        for (size_t i=0; i+1 < allocs.size(); i++) {
          if ( allocs[i+1].start - (allocs[i].start + allocs[i].length) >= blocksReq ) {
            allocs.insert( allocs.begin()+i+1 , AllocNode( allocs[i].start+allocs[i].length , blocksReq , std::move(label) ) );
            return getPtr(allocs[i+1].start);
          }
        }
        // Look for room after the last allocation.
        if ( nBlocks - (allocs.back().start + allocs.back().length) >= blocksReq ) {
          allocs.push_back( AllocNode( allocs.back().start + allocs.back().length , blocksReq , std::move(label) ) );
          return getPtr(allocs.back().start);
        }
      }

      Kokkos::abort( "The pool has run out of memory. Please initialize a larger pool." );
      return nullptr;
    };


    // Free an exact pointer previously returned by allocate(). Returns its block-rounded byte count.
    size_t free(void * ptr, std::string label = "") {
      std::lock_guard<std::recursive_mutex> lock(mutex);
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


    // Perform the same first-fit search as allocate() without changing allocator state.
    bool iGotRoom( size_t bytes ) const {
      std::lock_guard<std::recursive_mutex> lock(mutex);
      if (bytes == 0) return true;
      if constexpr (kokkos_debug) {
        if (!initialized()) Kokkos::abort("ERROR: querying an uninitialized LinearAllocator");
      }
      if (bytes > std::numeric_limits<size_t>::max()-(blockSize-1)) return false;
      size_t blocksReq = (bytes-1)/blockSize + 1; // Number of whole blocks needed for this allocation
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


    // Determine whether a pointer lies in the usable pool. Subtraction is evaluated only after ptr >= start to avoid
    // overflowing an end-address calculation.
    bool thisIsMyPointer(void * ptr_in) const {
      std::lock_guard<std::recursive_mutex> lock(mutex);
      if (pool == nullptr || ptr_in == nullptr) return false;
      std::uintptr_t ptr   = reinterpret_cast<uintptr_t>(ptr_in);
      std::uintptr_t start = reinterpret_cast<uintptr_t>(pool  );
      return ptr >= start && ptr-start < poolSize();
    }


    bool initialized() const {
      std::lock_guard<std::recursive_mutex> lock(mutex);
      return pool != nullptr;
    }


    size_t poolSize() const {
      std::lock_guard<std::recursive_mutex> lock(mutex);
      if (blockSize != 0 && nBlocks > std::numeric_limits<size_t>::max()/blockSize) {
        Kokkos::abort("ERROR: LinearAllocator pool-size overflow");
      }
      return nBlocks*blockSize;
    }


    size_t numAllocs() const {
      std::lock_guard<std::recursive_mutex> lock(mutex);
      return allocs.size();
    }


    // Transform a block index into a memory pointer
    void * getPtr( size_t blockIndex ) const {
      std::lock_guard<std::recursive_mutex> lock(mutex);
      if constexpr (kokkos_bounds_debug) {
        if (pool == nullptr || blockIndex >= nBlocks) Kokkos::abort("ERROR: LinearAllocator block index out of bounds");
      }
      return static_cast<char*>(pool) + blockIndex * blockSize;
    }


  };

}
