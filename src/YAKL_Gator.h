
#pragma once
// Included by YAKL.h

#include "YAKL_LinearAllocator.h"

namespace yakl {

  class Gator {
  protected:
    std::string                           pool_name;
    std::list<LinearAllocator>            pools;    // The pools managed by this class
    std::function<void *( size_t )>       mymalloc; // allocation function
    std::function<void( void * )>         myfree;   // free function
    std::function<void( void *, size_t )> myzero;   // zero function
    size_t growSize;   // Amount by which the pool grows in bytes
    size_t blockSize;  // Minimum allocation size
    std::string error_message_cannot_grow;
    std::string error_message_out_of_memory;

    std::mutex mtx;    // Internal mutex used to protect alloc and free in threaded regions

    void die(std::string str="") {
      std::cerr << str << std::endl;
      throw str;
    }


  public:

    Gator() {
    }


    // No moves allowed
    Gator            (      Gator && );
    Gator &operator= (      Gator && );
    Gator            (const Gator &  ) = delete;
    Gator &operator= (const Gator &  ) = delete;


    ~Gator() { finalize(); }


    // Initialize the pool allocator using environment variables and the passed malloc, free, and zero functions
    void init(std::function<void *( size_t )>       mymalloc  = [] (size_t bytes) -> void * { return ::malloc(bytes); },
              std::function<void( void * )>         myfree    = [] (void *ptr) { ::free(ptr); }                        ,
              std::function<void( void *, size_t )> myzero    = [] (void *ptr, size_t bytes) {}                        ,
              size_t initialSize                              = 1024*1024*1024                                         ,
              size_t growSize                                 = 1024*1024*1024                                         ,
              size_t blockSize                                = sizeof(size_t)                                         ,
              std::string pool_name                           = "Gator"                                                ,
              std::string error_message_out_of_memory         = ""                                                     ,
              std::string error_message_cannot_grow           = "" ) {
      this->mymalloc  = mymalloc ;
      this->myfree    = myfree   ;
      this->myzero    = myzero   ;
      this->growSize  = growSize ;
      this->blockSize = blockSize;
      this->pool_name = pool_name;
      this->error_message_out_of_memory = error_message_out_of_memory;
      this->error_message_cannot_grow   = error_message_cannot_grow  ;

      // Create the initial pool if the pool allocator is to be used
      pools.push_back( LinearAllocator( initialSize , blockSize , mymalloc , myfree , myzero ,
                                        pool_name , error_message_out_of_memory) );
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
          if (bytes > growSize) {
            std::cerr << "ERROR: For the pool allocator labeled " << pool_name << ":" << std::endl;
            std::cerr << "ERROR: Trying to allocate " << bytes << " bytes (" << bytes/1024./1024./1024. << " GB), "
                      << "but the current pool is too small, and growSize is only "
                      << growSize << " bytes (" << growSize/1024./1024./1024. << " GB). \nThus, the allocation will never fit in pool memory.\n";
            std::cerr << "This can happen for a number of reasons. \nCheck the size of the variable being allocated in the "
                      << "line above and see if it's what you expected. \nIf it's absurdly large, then you might have tried "
                      << "to pass in a negative value for the size, or the size got corrupted somehow. \nNOTE: If you compiled "
                      << "for the wrong GPU artchitecture, it sometimes shows up here as well. \nIf the size of the variable "
                      << "is realistic, then you should increase the initial pool size and probably the grow size as "
                      << "well. \nWhen individual variables consume sizable percentages of a pool, memory gets segmented, and "
                      << "the pool space isn't used efficiently. \nLarger pools will improve that. "
                      << "\nIn the extreme, you could create "
                      << "an initial pool that consumes most of the avialable memory. \nIf that still doesn't work, then "
                      << "it sounds like you're choosing a problem size that's too large for the number of compute "
                      << "nodes you're using.\n";
            std::cerr << error_message_cannot_grow << std::endl;
            printAllocsLeft();
            die();
          }
          pools.push_back( LinearAllocator( growSize , blockSize , mymalloc , myfree , myzero ,
                                            pool_name , error_message_out_of_memory) );
          ptr = pools.back().allocate(bytes,label);
        }
      }
      mtx.unlock();
      if (linear_bug) {
        std::cerr << "ERROR: For the pool allocator labeled " << pool_name << ":" << std::endl;
        die("ERROR: It looks like you've found a bug in LinearAllocator. Please report this at github.com/mrnorman/YAKL");
      }
      if (ptr != nullptr) {
        return ptr;
      } else {
        std::cerr << "ERROR: For the pool allocator labeled " << pool_name << ":" << std::endl;
        std::cerr << "Unable to allocate pointer. It looks like you might have run out of memory.";
        die( error_message_out_of_memory );
      }
      return nullptr;
    };


    // Free the specified pointer with the specified label
    void free(void *ptr , char const * label = "") {
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
      if (!pointer_valid) {
        std::cerr << "ERROR: For the pool allocator labeled " << pool_name << ":" << std::endl;
        std::cerr << "ERROR: Trying to free an invalid pointer\n";
        die("This means you have either already freed the pointer, or its address has been corrupted somehow.");
      }
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


