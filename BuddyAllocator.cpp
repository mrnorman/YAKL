/*
The BuddyAllocator.cpp files is courtesy of Mark Berrill
Mark Berrill
Oak Ridge Leadership Computing Facility
berrillma@ornl.gov
*/

#include "BuddyAllocator.h"

#include <cstring>
#include <iostream>
#include <utility>


// Fast ceil(log2(x))
static inline uint8_t log2( uint64_t x )
{
    uint8_t y = 0;
    while ( ( ( (uint64_t) 0x01 ) << y ) < x )
        y++;
    return y;
}
// Fast 2^n
static inline size_t pow2( uint8_t n ) { return ( (size_t) 0x01 ) << n; }


/********************************************************************
 * Constructor/destructor                                           *
 ********************************************************************/
BuddyAllocator::BuddyAllocator()
    : d_n( 0 ),
      d_log_N( 0 ),
      d_memory( nullptr ),
      d_size( nullptr ),
      d_N( nullptr ),
      d_ptr( nullptr )
{
}
BuddyAllocator::BuddyAllocator( size_t bytes, size_t blockSize,
    std::function<void*( size_t )> allocator, std::function<void( void* )> deallocator,
    std::function<void( void*, size_t )> zero )
    : d_n( 0 ),
      d_log_N( log2( blockSize ) ),
      d_memory( nullptr ),
      d_size( nullptr ),
      d_N( nullptr ),
      d_ptr( nullptr ),
      d_allocator( std::move( allocator ) ),
      d_deallocator( std::move( deallocator ) ),
      d_zero( std::move( zero ) )
{
    if ( pow2( d_log_N ) != blockSize )
        throw std::logic_error( "blockSize must be a power of 2" );
    // Compute the number of blocks to allocate (must be a power of 2)
    d_n = log2( bytes / blockSize );
    if ( blockSize * N_blocks() < bytes )
        d_n++;
    // Allocate the memory
    d_memory = d_allocator( N_bytes() );
    if ( !d_memory )
        throw std::bad_alloc();
    d_zero( d_memory, N_bytes() );
    // Initialize internal data
    allocateSize();
    for ( size_t i = 0; i < N_blocks(); i++ )
        setLevel( i, d_n + 1 );
    setLevel( 0, d_n );
    d_N   = new int[d_n + 1];
    d_ptr = new int[64 * ( d_n + 1 )];
    memset( d_N, 0, ( d_n + 1 ) * sizeof( int ) );
    memset( d_ptr, 0, 64 * ( d_n + 1 ) * sizeof( int ) );
    d_N[d_n]        = 1;
    d_ptr[64 * d_n] = 0;
}
BuddyAllocator::BuddyAllocator( BuddyAllocator&& rhs )
    : d_n( rhs.d_n ),
      d_log_N( rhs.d_log_N ),
      d_memory( rhs.d_memory ),
      d_size( rhs.d_size ),
      d_N( rhs.d_N ),
      d_ptr( rhs.d_ptr ),
      d_allocator( std::move( rhs.d_allocator ) ),
      d_deallocator( std::move( rhs.d_deallocator ) ),
      d_zero( std::move( rhs.d_zero ) )
{
    rhs.d_memory = nullptr;
    rhs.d_size   = nullptr;
    rhs.d_N      = nullptr;
    rhs.d_ptr    = nullptr;
}
BuddyAllocator& BuddyAllocator::operator=( BuddyAllocator&& rhs )
{
    if ( this == &rhs )
        return *this;
    std::swap( d_n, rhs.d_n );
    std::swap( d_log_N, rhs.d_log_N );
    std::swap( d_memory, rhs.d_memory );
    std::swap( d_size, rhs.d_size );
    std::swap( d_N, rhs.d_N );
    std::swap( d_ptr, rhs.d_ptr );
    std::swap( d_allocator, rhs.d_allocator );
    std::swap( d_deallocator, rhs.d_deallocator );
    std::swap( d_zero, rhs.d_zero );
    return *this;
}
BuddyAllocator::~BuddyAllocator()
{
    if ( d_memory != nullptr ) {
        if ( d_N[d_n] != 1 )
            std::cerr << "Some memory was not free'd before destroying BuddyAllocator\n";
        d_deallocator( d_memory );
        delete[] d_size;
        delete[] d_N;
        delete[] d_ptr;
    }
    d_memory = nullptr;
}


/********************************************************************
 * Allocate memory                                                   *
 ********************************************************************/
void* BuddyAllocator::allocate( size_t bytes )
{
    if ( bytes == 0 )
        return nullptr;
    // Get the number of blocks required (as 2^(n-1))
    int n = 0;
    while ( pow2( n + d_log_N ) < bytes )
        ++n;
    // Allocate the desired block
    int block = blockAlloc( n );
    if ( block == -1 )
        return nullptr;
    auto* ptr = (void*) ( (size_t) d_memory + ( (size_t) block << d_log_N ) );
    return ptr;
}


/********************************************************************
 * Free memory                                                       *
 ********************************************************************/
void BuddyAllocator::free( void* ptr )
{
    if ( ptr == nullptr )
        return;
    // Get the block id and number of blocks
    size_t block = ( (size_t) ptr - (size_t) d_memory ) >> d_log_N;
    if ( block >= N_blocks() )
        throw std::bad_alloc();
    int n = getLevel( block );
    if ( n < 0 )
        throw std::bad_alloc();
    // Zero the memory before returning it
    size_t bytes = pow2( n + d_log_N );
    d_zero( ptr, bytes );
    // Return the memory
    blockFree( n, block );
}


/********************************************************************
 * Allocate/free a block given the level                             *
 * Note: these functions must be thread-safe                         *
 ********************************************************************/
int BuddyAllocator::blockAlloc( int level )
{
    int block = -1;
    int& N    = d_N[level];
    int* ptr  = &d_ptr[64 * level];
    if ( N > 0 ) {
        block = ptr[--N];
        setLevel( block, level );
    } else {
        if ( level == d_n )
            return -1;
        block = blockAlloc( level + 1 );
        setLevel( block, level );
        ptr[N++] = block + pow2( level );
    }
    return block;
}
void BuddyAllocator::blockFree( int level, int block )
{
    int block2 = block ^ ( 0x01 << level );
    int k      = -1;
    int& N     = d_N[level];
    int* ptr   = &d_ptr[64 * level];
    for ( int i = 0; i < N; i++ ) {
        if ( ptr[i] == block2 )
            k = i;
    }
    if ( k != -1 ) {
        // Merge blocks and add to parent
        ptr[k] = ptr[--N];
        blockFree( level + 1, ( block & block2 ) );
    } else {
        // Add block to list
        if ( N == 64 )
            throw std::logic_error( "Internal error" );
        setLevel( block, d_n + 1 );
        ptr[N++] = block;
    }
}
