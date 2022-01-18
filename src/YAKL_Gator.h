
#pragma once

#include "YAKL_LinearAllocator.h"


class Gator {
protected:
  std::list<LinearAllocator> pools;               // The pools managed by this class
  std::function<void *( size_t )>       mymalloc; // allocation function
  std::function<void( void * )>         myfree;   // free function
  std::function<void( void *, size_t )> myzero;   // zero function
  size_t growSize;
  size_t blockSize;
  bool   enabled;

  std::mutex mtx;

  void die(std::string str="") {
    std::cerr << str << std::endl;
    throw str;
  }


public:

  Gator() {
    enabled = false;
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
    this->blockSize    = sizeof(size_t);

    enabled = true;
    char * env = std::getenv("GATOR_DISABLE");
    if ( env != nullptr ) {
      std::string resp(env);
      if (resp == "yes" || resp == "YES" || resp == "1" || resp == "true" || resp == "TRUE" || resp == "T") {
        enabled = false;
      }
    }

    // Check for GATOR_INITIAL_MB environment variable
    env = std::getenv("GATOR_INITIAL_MB");
    if ( env != nullptr ) {
      long int initial_mb = atol(env);
      if (initial_mb != 0) {
        initialSize = initial_mb*1024*1024;
        this->growSize = initialSize;
      } else {
        if (yakl::yakl_masterproc()) std::cout << "WARNING: Invalid GATOR_INITIAL_MB. Defaulting to 1GB\n";
      }
    }

    // Check for GATOR_GROW_MB environment variable
    env = std::getenv("GATOR_GROW_MB");
    if ( env != nullptr ) {
      long int grow_mb = atol(env);
      if (grow_mb != 0) {
        this->growSize = grow_mb*1024*1024;
      } else {
        if (yakl::yakl_masterproc()) std::cout << "WARNING: Invalid GATOR_GROW_MB. Defaulting to 1GB\n";
      }
    }

    // Check for GATOR_BLOCK_BYTES environment variable
    env = std::getenv("GATOR_BLOCK_BYTES");
    if ( env != nullptr ) {
      long int block_bytes = atol(env);
      if (block_bytes != 0 && block_bytes%sizeof(size_t) == 0) {
        this->blockSize = block_bytes;
      } else {
        if (yakl::yakl_masterproc()) std::cout << "WARNING: Invalid GATOR_BLOCK_BYTES. Defaulting to 128*sizeof(size_t)\n";
        if (yakl::yakl_masterproc()) std::cout << "         GATOR_BLOCK_BYTES must be > 0 and a multiple of sizeof(size_t)\n";
      }
    }

    if (enabled) {
      pools.push_back(LinearAllocator(initialSize , blockSize , mymalloc , myfree , myzero));
    }
  }


  void finalize() {
    if (enabled) {
      pools = std::list<LinearAllocator>();
    }
  }


  void printAllocsLeft() {
    for (auto it = pools.begin() ; it != pools.end() ; it++) {
      it->printAllocsLeft();
    }
  }


  void * allocate(size_t bytes, char const * label="") {
    #ifdef MEMORY_DEBUG
      if (yakl::yakl_masterproc()) std::cout << "MEMORY DEBUG: Gator attempting to allocate " << label << " with "
                                             << bytes << " bytes\n";
    #endif
    if (bytes == 0) return nullptr;
    // Loop through the pools and see if there's room. If so, allocate in one of them
    bool room_found = false;
    bool stacky_bug = false;
    void *ptr;
    mtx.lock();
    {
      for (auto it = pools.begin() ; it != pools.end() ; it++) {
        if (it->iGotRoom(bytes)) {
          ptr = it->allocate(bytes,label);
          room_found = true;
          if (ptr == nullptr) stacky_bug = true;
          break;
        }
      }
      if (!room_found) {
        #ifdef MEMORY_DEBUG
          if (yakl::yakl_masterproc()) std::cout << "MEMORY DEBUG: Current pools are not large enough. Adding a new pool of size "
                                                 << growSize << " bytes\n";
        #endif
        if (bytes > growSize) {
          std::cerr << "ERROR: Trying to allocate " << bytes << " bytes, but the current pool is too small, and growSize is only "
                    << growSize << " bytes. Thus, the allocation will never fit in pool memory.\n";
          die("You need to increase GATOR_GROW_MB and probably GATOR_INITIAL_MB as well\n");
        }
        pools.push_back( LinearAllocator(growSize , blockSize , mymalloc , myfree , myzero) );
        ptr = pools.back().allocate(bytes,label);
      }
    }
    mtx.unlock();
    if (stacky_bug) die("It looks like there might be a bug in LinearAllocator. Please report this at github.com/mrnorman/YAKL");
    if (ptr != nullptr) {
      return ptr;
    } else {
      die("Unable to allocate pointer. It looks like you might have run out of memory.");
    }
    return nullptr;
  };


  void free(void *ptr , char const * label = "") {
    #ifdef MEMORY_DEBUG
      if (yakl::yakl_masterproc()) std::cout << "MEMORY DEBUG: Gator attempting to free " << label
                                             << " with the pointer: " << ptr << "\n";
    #endif
    bool pointer_valid = false;
    mtx.lock();
    {
      // Iterate backwards. It's assumed accesses are stack-like
      for (auto it = pools.rbegin() ; it != pools.rend() ; it++) {
        if (it->thisIsMyPointer(ptr)) {
          it->free(ptr,label);
          pointer_valid = true;
          break;
        }
      }
    }
    mtx.unlock();
    if (!pointer_valid) die("Error: Trying to free an invalid pointer");
  };


  size_t poolSize( ) const {
    size_t sz = 0;
    for (auto it = pools.begin() ; it != pools.end() ; it++) { sz += it->poolSize(); }
    return sz;
  }


  size_t numAllocs( ) const {
    size_t allocs = 0;
    for (auto it = pools.begin() ; it != pools.end() ; it++) { allocs += it->numAllocs(); }
    return allocs;
  }

};


