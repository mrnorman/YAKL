
#pragma once
// Included by YAKL_Gator.h

#include "YAKL_LinearAllocator.h"

namespace yakl {

  class Gator {
  protected:
    std::list<LinearAllocator> pools;               // The pools managed by this class
    std::function<void *( size_t )>       mymalloc; // allocation function
    std::function<void( void * )>         myfree;   // free function
    std::function<void( void *, size_t )> myzero;   // zero function
    size_t growSize;   // Amount by which the pool grows in bytes
    size_t blockSize;  // Minimum allocation size

    std::mutex mtx;    // Internal mutex used to protect alloc and free in threaded regions

    void die(std::string str="") {
      std::cerr << str << std::endl;
      throw str;
    }


  public:

    Gator() {
    }


    // No copies or moves allowed
    Gator            (      Gator && );
    Gator &operator= (      Gator && );
    Gator            (const Gator &  ) = delete;
    Gator &operator= (const Gator &  ) = delete;


    ~Gator() { finalize(); }


    // Initialize the pool allocator using environment variables and the passed malloc, free, and zero functions
    void init( std::function<void *( size_t )>       mymalloc  = [] (size_t bytes) -> void * { return ::malloc(bytes); } ,
               std::function<void( void * )>         myfree    = [] (void *ptr) { ::free(ptr); }                         ,
               std::function<void( void *, size_t )> myzero    = [] (void *ptr, size_t bytes) {}                         ,
               size_t initialSize = 1024*1024*1024                                                                       ,
               size_t growSize    = 1024*1024*1024                                                                       ,
               size_t blockSize   = sizeof(size_t) ) {
      this->mymalloc  = mymalloc ;
      this->myfree    = myfree   ;
      this->myzero    = myzero   ;
      this->growSize  = growSize ;
      this->blockSize = blockSize;

      // Create the initial pool if the pool allocator is to be used
      pools.push_back( LinearAllocator(initialSize , blockSize , mymalloc , myfree , myzero) );
    }


    void finalize() { pools = std::list<LinearAllocator>(); }


    void printAllocsLeft() {
      // Used for debugging mainly. Prints all existing allocations
      for (auto it = pools.begin() ; it != pools.end() ; it++) {
        it->printAllocsLeft();
      }
    }


    // Allocate memory with the specified number of bytes and the specified label
    void * allocate(size_t bytes, char const * label="") {
      #ifdef MEMORY_DEBUG
        if (yakl::yakl_mainproc()) std::cout << "MEMORY DEBUG: Gator attempting to allocate " << label << " with "
                                             << bytes << " bytes\n";
      #endif
      if (bytes == 0) return nullptr;
      // Loop through the pools and see if there's room. If so, allocate in one of them
      bool room_found = false;  // Whether room exists for the allocation
      bool linear_bug = false;  // Whether there's an apparent bug in the LinearAllocator allocate() function
      void *ptr;                // Allocated pointer
      // Protect against multiple threads trying to allocate at the same time
      mtx.lock();
      {
        // Start at the first pool, see if it has room.
        // If so, allocate in that pool and break. If not, try the next pool.
        for (auto it = pools.begin() ; it != pools.end() ; it++) {
          if (it->iGotRoom(bytes)) {
            ptr = it->allocate(bytes,label);
            room_found = true;
            if (ptr == nullptr) linear_bug = true;
            break;
          }
        }
        // If you've gone through all of the existing pools, and room hasn't been found, then it's time to add a new pool
        if (!room_found) {
          #ifdef MEMORY_DEBUG
            if (yakl::yakl_mainproc()) std::cout << "MEMORY DEBUG: Current pools are not large enough. "
                                                 << "Adding a new pool of size "
                                                 << growSize << " bytes\n";
          #endif
          if (bytes > growSize) {
            std::cerr << "ERROR: Trying to allocate " << bytes
                      << " bytes, but the current pool is too small, and growSize is only "
                      << growSize << " bytes. Thus, the allocation will never fit in pool memory.\n";
            die("You need to increase GATOR_GROW_MB and probably GATOR_INITIAL_MB as well\n");
          }
          pools.push_back( LinearAllocator(growSize , blockSize , mymalloc , myfree , myzero) );
          ptr = pools.back().allocate(bytes,label);
        }
      }
      mtx.unlock();
      if (linear_bug) {
        die("It looks like there might be a bug in LinearAllocator. Please report this at github.com/mrnorman/YAKL");
      }
      if (ptr != nullptr) {
        return ptr;
      } else {
        die("Unable to allocate pointer. It looks like you might have run out of memory.");
      }
      return nullptr;
    };


    // Free the specified pointer with the specified label
    void free(void *ptr , char const * label = "") {
      #ifdef MEMORY_DEBUG
        if (yakl::yakl_mainproc()) std::cout << "MEMORY DEBUG: Gator attempting to free " << label
                                             << " with the pointer: " << ptr << "\n";
      #endif
      bool pointer_valid = false;
      // Protect against multiple threads trying to free at the same time
      mtx.lock();
      {
        // Go through each pool. If the pointer lives in that pool, then free it.
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


    // Get the total size of all of the pools put together
    size_t poolSize( ) const {
      size_t sz = 0;
      for (auto it = pools.begin() ; it != pools.end() ; it++) { sz += it->poolSize(); }
      return sz;
    }


    // Get the total number of allocations in all of the pools put together
    size_t numAllocs( ) const {
      size_t allocs = 0;
      for (auto it = pools.begin() ; it != pools.end() ; it++) { allocs += it->numAllocs(); }
      return allocs;
    }

  };

}


