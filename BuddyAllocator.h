/*
The BuddyAllocator.h files is courtesy of Mark Berrill
Mark Berrill
Oak Ridge Leadership Computing Facility
berrillma@ornl.gov
*/

#ifndef included_BuddyAllocator
#define included_BuddyAllocator


#include <functional>
#include <stdlib.h>


/** \class BuddyAllocator
 *
 * This class provides basic routines to allocate/deallocate memory
 */
class BuddyAllocator
{
public:
    //! Default constructor
    BuddyAllocator( size_t bytes, size_t blockSize = 1024,
        std::function<void *( size_t )> allocator  = ::malloc,
        std::function<void( void * )> deallocator  = ::free,
        std::function<void( void *, size_t )> zero = []( void *, size_t ) {} );

    //! Empty constructor
    BuddyAllocator();

    //! Destructor
    ~BuddyAllocator();

    // Copy/assignment constructors
    BuddyAllocator( BuddyAllocator && );
    BuddyAllocator &operator                 =( BuddyAllocator && );
    BuddyAllocator( const BuddyAllocator & ) = delete;
    BuddyAllocator &operator=( const BuddyAllocator & ) = delete;

    //! Return the classname
    static constexpr const char *classname() { return "BuddyAllocator"; }

    //! Allocate memory
    void *allocate( size_t bytes );

    //! Free memory
    void free( void *ptr );


private:
    uint8_t d_n;     // Number of blocks (log2)
    uint8_t d_log_N; // Size of a block (log2)
    void *d_memory;  // Raw memory pointer

    // Functions to get/clear the level id of the block
    int8_t *d_size; // Data to hold the level index
    inline void allocateSize() { d_size = new int8_t[N_blocks()]; }
    inline void setLevel( int block, int level ) { d_size[block] = level; }
    inline int getLevel( int block ) { return d_size[block]; }

    // Functions to allocate/free a block from a level (need to be thread-safe)
    int *d_N;
    int *d_ptr;
    int blockAlloc( int level );
    void blockFree( int level, int block );

    // Functions to allocate/free/zero memory
    std::function<void *( size_t )> d_allocator;
    std::function<void( void * )> d_deallocator;
    std::function<void( void *, size_t )> d_zero;

    // Helper functions
    inline size_t N_blocks() const { return ( (size_t) 0x01 ) << d_n; }
    inline size_t N_bytes() const { return ( (size_t) 0x01 ) << ( d_n + d_log_N ); }
};


#endif
