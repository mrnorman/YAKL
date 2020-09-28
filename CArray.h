
#pragma once

template <class T, int rank, int myMem> class Array<T,rank,myMem,styleC> {
public:

  index_t offsets  [rank];  // Precomputed dimension offsets for efficient data access into a 1-D pointer
  index_t dimension[rank];  // Sizes of the 8 possible dimensions
  T       * myData;         // Pointer to the flattened internal data
  int     * refCount;       // Pointer shared by multiple copies of this Array to keep track of allcation / free
  bool    owned;            // Whether is is owned (owned = allocated,ref_counted,deallocated) or not
  #ifdef YAKL_DEBUG
    std::string myname;     // Label for debug printing. Only stored if debugging is turned on
  #endif


  // Start off all constructors making sure the pointers are null
  YAKL_INLINE void nullify() {
    owned    = true;
    myData   = nullptr;
    refCount = nullptr;
  }

  /* CONSTRUCTORS
  Always nullify before beginning so that myData == nullptr upon init.
  */
  YAKL_INLINE Array() {
    nullify();
  }
  YAKL_INLINE Array(char const * label) {
    nullify();
    #ifdef YAKL_DEBUG
      myname = std::string(label);
    #endif
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Owned constructors
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Array(char const * label, index_t const d1) {
    #ifdef YAKL_DEBUG
      if( rank != 1 ) { yakl_throw("ERROR: Calling invalid constructor on rank 1 Array"); }
      myname = std::string(label);
    #endif
    nullify();
    deallocate();
    dimension[0] = d1;
    compute_offsets();
    allocate(label);
  }
  Array(char const * label, index_t const d1, index_t const d2) {
    #ifdef YAKL_DEBUG
      if( rank != 2 ) { yakl_throw("ERROR: Calling invalid constructor on rank 2 Array"); }
      myname = std::string(label);
    #endif
    nullify();
    deallocate();
    dimension[0] = d1;
    dimension[1] = d2;
    compute_offsets();
    allocate(label);
  }
  Array(char const * label, index_t const d1, index_t const d2, index_t const d3) {
    #ifdef YAKL_DEBUG
      if( rank != 3 ) { yakl_throw("ERROR: Calling invalid constructor on rank 3 Array"); }
      myname = std::string(label);
    #endif
    nullify();
    deallocate();
    dimension[0] = d1;
    dimension[1] = d2;
    dimension[2] = d3;
    compute_offsets();
    allocate(label);
  }
  Array(char const * label, index_t const d1, index_t const d2, index_t const d3, index_t const d4) {
    #ifdef YAKL_DEBUG
      if( rank != 4 ) { yakl_throw("ERROR: Calling invalid constructor on rank 4 Array"); }
      myname = std::string(label);
    #endif
    nullify();
    deallocate();
    dimension[0] = d1;
    dimension[1] = d2;
    dimension[2] = d3;
    dimension[3] = d4;
    compute_offsets();
    allocate(label);
  }
  Array(char const * label, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5) {
    #ifdef YAKL_DEBUG
      if( rank != 5 ) { yakl_throw("ERROR: Calling invalid constructor on rank 5 Array"); }
      myname = std::string(label);
    #endif
    nullify();
    deallocate();
    dimension[0] = d1;
    dimension[1] = d2;
    dimension[2] = d3;
    dimension[3] = d4;
    dimension[4] = d5;
    compute_offsets();
    allocate(label);
  }
  Array(char const * label, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5, index_t const d6) {
    #ifdef YAKL_DEBUG
      if( rank != 6 ) { yakl_throw("ERROR: Calling invalid constructor on rank 6 Array"); }
      myname = std::string(label);
    #endif
    nullify();
    deallocate();
    dimension[0] = d1;
    dimension[1] = d2;
    dimension[2] = d3;
    dimension[3] = d4;
    dimension[4] = d5;
    dimension[5] = d6;
    compute_offsets();
    allocate(label);
  }
  Array(char const * label, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5, index_t const d6, index_t const d7) {
    #ifdef YAKL_DEBUG
      if( rank != 7 ) { yakl_throw("ERROR: Calling invalid constructor on rank 7 Array"); }
      myname = std::string(label);
    #endif
    nullify();
    deallocate();
    dimension[0] = d1;
    dimension[1] = d2;
    dimension[2] = d3;
    dimension[3] = d4;
    dimension[4] = d5;
    dimension[5] = d6;
    dimension[6] = d7;
    compute_offsets();
    allocate(label);
  }
  Array(char const * label, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5, index_t const d6, index_t const d7, index_t const d8) {
    #ifdef YAKL_DEBUG
      if( rank != 8 ) { yakl_throw("ERROR: Calling invalid constructor on rank 8 Array"); }
      myname = std::string(label);
    #endif
    nullify();
    deallocate();
    dimension[0] = d1;
    dimension[1] = d2;
    dimension[2] = d3;
    dimension[3] = d4;
    dimension[4] = d5;
    dimension[5] = d6;
    dimension[6] = d7;
    dimension[7] = d8;
    compute_offsets();
    allocate(label);
  }
  template <class INT, typename std::enable_if< std::is_integral<INT>::value , int >::type = 0>
  Array(char const * label, std::vector<INT> const dims) {
    #ifdef YAKL_DEBUG
      if ( dims.size() < rank ) { yakl_throw("ERROR: dims < rank"); }
      if ( rank < 1 || rank > 8 ) { yakl_throw("ERROR: Invalid rank, must be between 1 and 8"); }
      myname = std::string(label);
    #endif
    nullify();
    deallocate();
    for (int i=0; i < rank; i++) {
      dimension[i] = dims[i];
    }
    compute_offsets();
    allocate(label);
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Non-owned constructors
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Array(char const * label, T * data, index_t const d1) {
    #ifdef YAKL_DEBUG
      if( rank != 1 ) { yakl_throw("ERROR: Calling invalid constructor on rank 1 Array"); }
      myname = std::string(label);
    #endif
    nullify();
    owned = false;
    dimension[0] = d1;
    compute_offsets();
    myData = data;
  }
  Array(char const * label, T * data, index_t const d1, index_t const d2) {
    #ifdef YAKL_DEBUG
      if( rank != 2 ) { yakl_throw("ERROR: Calling invalid constructor on rank 2 Array"); }
      myname = std::string(label);
    #endif
    nullify();
    owned = false;
    dimension[0] = d1;
    dimension[1] = d2;
    compute_offsets();
    myData = data;
  }
  Array(char const * label, T * data, index_t const d1, index_t const d2, index_t const d3) {
    #ifdef YAKL_DEBUG
      if( rank != 3 ) { yakl_throw("ERROR: Calling invalid constructor on rank 3 Array"); }
      myname = std::string(label);
    #endif
    nullify();
    owned = false;
    dimension[0] = d1;
    dimension[1] = d2;
    dimension[2] = d3;
    compute_offsets();
    myData = data;
  }
  Array(char const * label, T * data, index_t const d1, index_t const d2, index_t const d3, index_t const d4) {
    #ifdef YAKL_DEBUG
      if( rank != 4 ) { yakl_throw("ERROR: Calling invalid constructor on rank 4 Array"); }
      myname = std::string(label);
    #endif
    nullify();
    owned = false;
    dimension[0] = d1;
    dimension[1] = d2;
    dimension[2] = d3;
    dimension[3] = d4;
    compute_offsets();
    myData = data;
  }
  Array(char const * label, T * data, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5) {
    #ifdef YAKL_DEBUG
      if( rank != 5 ) { yakl_throw("ERROR: Calling invalid constructor on rank 5 Array"); }
      myname = std::string(label);
    #endif
    nullify();
    owned = false;
    dimension[0] = d1;
    dimension[1] = d2;
    dimension[2] = d3;
    dimension[3] = d4;
    dimension[4] = d5;
    compute_offsets();
    myData = data;
  }
  Array(char const * label, T * data, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5, index_t const d6) {
    #ifdef YAKL_DEBUG
      if( rank != 6 ) { yakl_throw("ERROR: Calling invalid constructor on rank 6 Array"); }
      myname = std::string(label);
    #endif
    nullify();
    owned = false;
    dimension[0] = d1;
    dimension[1] = d2;
    dimension[2] = d3;
    dimension[3] = d4;
    dimension[4] = d5;
    dimension[5] = d6;
    compute_offsets();
    myData = data;
  }
  Array(char const * label, T * data, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5, index_t const d6, index_t const d7) {
    #ifdef YAKL_DEBUG
      if( rank != 7 ) { yakl_throw("ERROR: Calling invalid constructor on rank 7 Array"); }
      myname = std::string(label);
    #endif
    nullify();
    owned = false;
    dimension[0] = d1;
    dimension[1] = d2;
    dimension[2] = d3;
    dimension[3] = d4;
    dimension[4] = d5;
    dimension[5] = d6;
    dimension[6] = d7;
    compute_offsets();
    myData = data;
  }
  Array(char const * label, T * data, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5, index_t const d6, index_t const d7, index_t const d8) {
    #ifdef YAKL_DEBUG
      if( rank != 8 ) { yakl_throw("ERROR: Calling invalid constructor on rank 8 Array"); }
      myname = std::string(label);
    #endif
    nullify();
    owned = false;
    dimension[0] = d1;
    dimension[1] = d2;
    dimension[2] = d3;
    dimension[3] = d4;
    dimension[4] = d5;
    dimension[5] = d6;
    dimension[6] = d7;
    dimension[7] = d8;
    compute_offsets();
    myData = data;
  }
  template <class INT, typename std::enable_if< std::is_integral<INT>::value , int >::type = 0>
  Array(char const * label, T * data, std::vector<INT> const dims) {
    #ifdef YAKL_DEBUG
      if ( dims.size() < rank ) { yakl_throw("ERROR: dims < rank"); }
      if ( rank < 1 || rank > 8 ) { yakl_throw("ERROR: Invalid rank, must be between 1 and 8"); }
      myname = std::string(label);
    #endif
    nullify();
    owned = false;
    for (int i=0; i < rank; i++) {
      dimension[i] = dims[i];
    }
    compute_offsets();
    myData = data;
  }


  /*
  COPY CONSTRUCTORS / FUNCTIONS
  This shares the pointers with another Array and increments the refCounter
  */
  Array(Array const &rhs) {
    // constructor, so no need to deallocate
    nullify();
    owned    = rhs.owned;
    for (int i=0; i<rank; i++) {
      offsets  [i] = rhs.offsets  [i];
      dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      myname = rhs.myname;
    #endif
    myData   = rhs.myData;
    refCount = rhs.refCount;
    if (owned) { (*refCount)++; }
  }


  Array & operator=(Array const &rhs) {
    if (this == &rhs) {
      return *this;
    }
    owned    = rhs.owned;
    deallocate();
    for (int i=0; i<rank; i++) {
      offsets  [i] = rhs.offsets  [i];
      dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      myname = rhs.myname;
    #endif
    myData   = rhs.myData;
    refCount = rhs.refCount;
    if (owned) { (*refCount)++; }

    return *this;
  }


  /*
  MOVE CONSTRUCTORS
  This steals the pointers form the rhs rather than sharing and sets rhs pointers to nullptr.
  Therefore, no need to increment refCout
  */
  Array(Array &&rhs) {
    // constructor, so no need to deallocate
    nullify();
    owned    = rhs.owned;
    for (int i=0; i<rank; i++) {
      offsets  [i] = rhs.offsets  [i];
      dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      myname = rhs.myname;
    #endif
    myData   = rhs.myData;
    refCount = rhs.refCount;

    rhs.myData   = nullptr;
    rhs.refCount = nullptr;
  }


  Array& operator=(Array &&rhs) {
    if (this == &rhs) { return *this; }
    owned    = rhs.owned;
    deallocate();
    for (int i=0; i<rank; i++) {
      offsets  [i] = rhs.offsets  [i];
      dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      myname = rhs.myname;
    #endif
    myData   = rhs.myData;
    refCount = rhs.refCount;

    rhs.myData   = nullptr;
    rhs.refCount = nullptr;

    return *this;
  }


  /*
  DESTRUCTOR
  Decrement the refCounter, and if it's zero, deallocate and nullify.  
  */
  YAKL_INLINE ~Array() {
    deallocate();
  }


  /* ARRAY INDEXERS (FORTRAN index ordering)
  Return the element at the given index (either read-only or read-write)
  */
  YAKL_INLINE T &operator()(index_t const i0) const {
    #ifdef YAKL_DEBUG
      if ( rank != 1 ) { yakl_throw("ERROR: Calling invalid function on rank 1 Array"); }
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
    #endif
    index_t ind = i0;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(index_t const i0, index_t const i1) const {
    #ifdef YAKL_DEBUG
      if ( rank != 2 ) { yakl_throw("ERROR: Calling invalid function on rank 2 Array"); }
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,0,dimension[1]-1,__FILE__,__LINE__);
    #endif
    index_t ind = i0*offsets[0] + i1;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(index_t const i0, index_t const i1, index_t const i2) const {
    #ifdef YAKL_DEBUG
      if ( rank != 3 ) { yakl_throw("ERROR: Calling invalid function on rank 3 Array"); }
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,0,dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,0,dimension[2]-1,__FILE__,__LINE__);
    #endif
    index_t ind = i0*offsets[0] + i1*offsets[1] + i2;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(index_t const i0, index_t const i1, index_t const i2, index_t const i3) const {
    #ifdef YAKL_DEBUG
      if ( rank != 4 ) { yakl_throw("ERROR: Calling invalid function on rank 4 Array"); }
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,0,dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,0,dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,0,dimension[3]-1,__FILE__,__LINE__);
    #endif
    index_t ind = i0*offsets[0] + i1*offsets[1] + i2*offsets[2] + i3;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(index_t const i0, index_t const i1, index_t const i2, index_t const i3, index_t const i4) const {
    #ifdef YAKL_DEBUG
      if ( rank != 5 ) { yakl_throw("ERROR: Calling invalid function on rank 5 Array"); }
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,0,dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,0,dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,0,dimension[3]-1,__FILE__,__LINE__);
      this->check_index(4,i4,0,dimension[4]-1,__FILE__,__LINE__);
    #endif
    index_t ind = i0*offsets[0] + i1*offsets[1] + i2*offsets[2] + i3*offsets[3] + i4;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(index_t const i0, index_t const i1, index_t const i2, index_t const i3, index_t const i4, index_t const i5) const {
    #ifdef YAKL_DEBUG
      if ( rank != 6 ) { yakl_throw("ERROR: Calling invalid function on rank 6 Array"); }
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,0,dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,0,dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,0,dimension[3]-1,__FILE__,__LINE__);
      this->check_index(4,i4,0,dimension[4]-1,__FILE__,__LINE__);
      this->check_index(5,i5,0,dimension[5]-1,__FILE__,__LINE__);
    #endif
    index_t ind = i0*offsets[0] + i1*offsets[1] + i2*offsets[2] + i3*offsets[3] + i4*offsets[4] + i5;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(index_t const i0, index_t const i1, index_t const i2, index_t const i3, index_t const i4, index_t const i5, index_t const i6) const {
    #ifdef YAKL_DEBUG
      if ( rank != 7 ) { yakl_throw("ERROR: Calling invalid function on rank 7 Array"); }
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,0,dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,0,dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,0,dimension[3]-1,__FILE__,__LINE__);
      this->check_index(4,i4,0,dimension[4]-1,__FILE__,__LINE__);
      this->check_index(5,i5,0,dimension[5]-1,__FILE__,__LINE__);
      this->check_index(6,i6,0,dimension[6]-1,__FILE__,__LINE__);
    #endif
    index_t ind = i0*offsets[0] + i1*offsets[1] + i2*offsets[2] + i3*offsets[3] + i4*offsets[4] + i5*offsets[5] + i6;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(index_t const i0, index_t const i1, index_t const i2, index_t const i3, index_t const i4, index_t const i5, index_t const i6, index_t const i7) const {
    #ifdef YAKL_DEBUG
      if ( rank != 8 ) { yakl_throw("ERROR: Calling invalid function on rank 8 Array"); }
      this->check_index(0,i0,0,dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,0,dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,0,dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,0,dimension[3]-1,__FILE__,__LINE__);
      this->check_index(4,i4,0,dimension[4]-1,__FILE__,__LINE__);
      this->check_index(5,i5,0,dimension[5]-1,__FILE__,__LINE__);
      this->check_index(6,i6,0,dimension[6]-1,__FILE__,__LINE__);
      this->check_index(7,i7,0,dimension[7]-1,__FILE__,__LINE__);
    #endif
    index_t ind = i0*offsets[0] + i1*offsets[1] + i2*offsets[2] + i3*offsets[3] + i4*offsets[4] + i5*offsets[5] + i6*offsets[6] + i7;
    return myData[ind];
  }


  inline void check_index(int const dim, long const ind, long const lb, long const ub, char const *file, int const line) const {
    if (ind < lb || ind > ub) {
      #ifdef YAKL_DEBUG
        std::cout << "For Array labeled: " << myname << "\n";
      #endif
      std::cout << "Index " << dim+1 << " of " << rank << " out of bounds\n";
      std::cout << "File, Line: " << file << ", " << line << "\n";
      std::cout << "Index: " << ind << ". Bounds: (" << lb << "," << ub << ")\n";
      throw "";
    }
  }


  template <int N> YAKL_INLINE void slice( Dims const &dims , Array<T,N,myMem,styleC> &store ) const {
    #ifdef YAKL_DEBUG
      if (rank != dims.size()) {
        yakl_throw( "ERROR: rank must be equal to dims.size()" );
      }
    #endif
    store.owned = false;
    for (int i = rank-1; i > rank-1-N; i--) {
      store.dimension[i-(rank-N)] = dimension[i];
      store.offsets  [i-(rank-N)] = offsets  [i];
    }
    index_t retOff = 0;
    for (int i = rank-1-N; i >= 0; i--) {
      retOff += dims.data[i]*offsets[i];
    }
    store.myData = &(this->myData[retOff]);
  }
  template <int N> YAKL_INLINE void slice( int i0 , Array<T,N,myMem,styleC> &store ) const {
    slice( {i0} , store );
  }
  template <int N> YAKL_INLINE void slice( int i0, int i1 , Array<T,N,myMem,styleC> &store ) const {
    slice( {i0,i1} , store );
  }
  template <int N> YAKL_INLINE void slice( int i0, int i1, int i2, Array<T,N,myMem,styleC> &store ) const {
    slice( {i0,i1,i2} , store );
  }
  template <int N> YAKL_INLINE void slice( int i0, int i1, int i2, int i3, Array<T,N,myMem,styleC> &store ) const {
    slice( {i0,i1,i2,i3} , store );
  }
  template <int N> YAKL_INLINE void slice( int i0, int i1, int i2, int i3, int i4, Array<T,N,myMem,styleC> &store ) const {
    slice( {i0,i1,i2,i3,i4} , store );
  }
  template <int N> YAKL_INLINE void slice( int i0, int i1, int i2, int i3, int i4, int i5, Array<T,N,myMem,styleC> &store ) const {
    slice( {i0,i1,i2,i3,i4,i5} , store );
  }
  template <int N> YAKL_INLINE void slice( int i0, int i1, int i2, int i3, int i4, int i5, int i6, Array<T,N,myMem,styleC> &store ) const {
    slice( {i0,i1,i2,i3,i4,i5,i6} , store );
  }
  template <int N> YAKL_INLINE void slice( int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, Array<T,N,myMem,styleC> &store ) const {
    slice( {i0,i1,i2,i3,i4,i5,i6,i7} , store );
  }


  template <int N> YAKL_INLINE Array<T,N,myMem,styleC> slice( Dims const &dims ) const {
    Array<T,N,myMem,styleC> ret;
    slice( dims , ret );
    return ret;
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleC> slice( int i0 ) const {
    return slice<N>( {i0} );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleC> slice( int i0, int i1 ) const {
    return slice<N>( {i0,i1} );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleC> slice( int i0, int i1, int i2 ) const {
    return slice<N>( {i0,i1,i2} );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleC> slice( int i0, int i1, int i2, int i3 ) const {
    return slice<N>( {i0,i1,i2,i3} );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleC> slice( int i0, int i1, int i2, int i3, int i4 ) const {
    return slice<N>( {i0,i1,i2,i3,i4} );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleC> slice( int i0, int i1, int i2, int i3, int i4, int i5 ) const {
    return slice<N>( {i0,i1,i2,i3,i4,i5} );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleC> slice( int i0, int i1, int i2, int i3, int i4, int i5, int i6 ) const {
    return slice<N>( {i0,i1,i2,i3,i4,i5,i6} );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleC> slice( int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7 ) const {
    return slice<N>( {i0,i1,i2,i3,i4,i5,i6,i7} );
  }


  inline Array<T,rank,memHost,styleC> createHostCopy() const {
    Array<T,rank,memHost,styleC> ret;  // nullified + owned == true
    for (int i=0; i<rank; i++) {
      ret.offsets  [i] = offsets  [i];
      ret.dimension[i] = dimension[i];
    }
    #ifdef YAKL_DEBUG
      ret.myname = myname;
    #endif
    ret.allocate();
    if (myMem == memHost) {
      for (index_t i=0; i<totElems(); i++) { ret.myData[i] = myData[i]; }
    } else {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),cudaMemcpyDeviceToHost,0);
        check_last_error();
        cudaDeviceSynchronize();
        check_last_error();
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),hipMemcpyDeviceToHost,0);
        check_last_error();
        hipDeviceSynchronize();
        check_last_error();
      #else
        for (index_t i=0; i<totElems(); i++) { ret.myData[i] = myData[i]; }
      #endif
    }
    return ret;
  }


  inline Array<T,rank,memDevice,styleC> createDeviceCopy() const {
    Array<T,rank,memDevice,styleC> ret;  // nullified + owned == true
    for (int i=0; i<rank; i++) {
      ret.offsets  [i] = offsets  [i];
      ret.dimension[i] = dimension[i];
    }
    #ifdef YAKL_DEBUG
      ret.myname = myname;
    #endif
    ret.allocate();
    if (myMem == memHost) {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),cudaMemcpyHostToDevice,0);
        check_last_error();
        cudaDeviceSynchronize();
        check_last_error();
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),hipMemcpyHostToDevice,0);
        check_last_error();
        hipDeviceSynchronize();
        check_last_error();
      #else
        for (index_t i=0; i<totElems(); i++) { ret.myData[i] = myData[i]; }
      #endif
    } else {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),cudaMemcpyDeviceToDevice,0);
        check_last_error();
        cudaDeviceSynchronize();
        check_last_error();
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),hipMemcpyDeviceToDevice,0);
        check_last_error();
        hipDeviceSynchronize();
        check_last_error();
      #else
        for (index_t i=0; i<totElems(); i++) { ret.myData[i] = myData[i]; }
      #endif
    }
    return ret;
  }


  inline void deep_copy_to(Array<T,rank,memHost,styleC> lhs) {
    if (myMem == memHost) {
      for (index_t i=0; i<totElems(); i++) { lhs.myData[i] = myData[i]; }
    } else {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),cudaMemcpyDeviceToHost,0);
        check_last_error();
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),hipMemcpyDeviceToHost,0);
        check_last_error();
      #else
        for (index_t i=0; i<totElems(); i++) { lhs.myData[i] = myData[i]; }
      #endif
    }
  }


  inline void deep_copy_to(Array<T,rank,memDevice,styleC> lhs) {
    if (myMem == memHost) {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),cudaMemcpyHostToDevice,0);
        check_last_error();
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),hipMemcpyHostToDevice,0);
        check_last_error();
      #else
        for (index_t i=0; i<totElems(); i++) { lhs.myData[i] = myData[i]; }
      #endif
    } else {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),cudaMemcpyDeviceToDevice,0);
        check_last_error();
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),hipMemcpyDeviceToDevice,0);
        check_last_error();
      #else
        for (index_t i=0; i<totElems(); i++) { lhs.myData[i] = myData[i]; }
      #endif
    }
  }


  /* ACCESSORS */
  YAKL_INLINE int get_rank() const {
    return rank;
  }
  YAKL_INLINE index_t get_totElems() const {
    index_t tot = dimension[0];
    for (int i=1; i<rank; i++) { tot *= dimension[i]; }
    return tot;
  }
  YAKL_INLINE index_t get_elem_count() const {
    return get_totElems();
  }
  YAKL_INLINE index_t totElems() const {
    return get_totElems();
  }
  YAKL_INLINE SArray<index_t,1,rank> get_dimensions() const {
    SArray<index_t,1,rank> ret;
    for (int i=0; i<rank; i++) { ret(i) = dimension[i]; }
    return ret;
  }
  YAKL_INLINE T *data() const {
    return myData;
  }
  YAKL_INLINE T *get_data() const {
    return myData;
  }
  YAKL_INLINE index_t extent( int const dim ) const {
    return dimension[dim];
  }
  YAKL_INLINE bool span_is_contiguous() const {
    return true;
  }
  YAKL_INLINE int use_count() const {
    if (owned) {
      return *refCount;
    } else {
      return -1;
    }
  }
  YAKL_INLINE bool initialized() const {
    return myData != nullptr;
  }
  const char* label() const {
    #ifdef YAKL_DEBUG
      return myname.c_str();
    #else
      return "";
    #endif
  }


  /* OPERATOR<<
  Print the array. If it's 2-D, print a pretty looking matrix */
  inline friend std::ostream &operator<<(std::ostream& os, Array const &v) {
    #ifdef YAKL_DEBUG
      os << "For Array labeled: " << v.myname << "\n";
    #endif
    os << "Number of Dimensions: " << rank << "\n";
    os << "Total Number of Elements: " << v.totElems() << "\n";
    os << "Dimension Sizes: ";
    for (int i=0; i<rank; i++) {
      os << v.dimension[i] << ", ";
    }
    os << "\n";
    for (index_t i=0; i<v.totElems(); i++) {
      os << v.myData[i] << " ";
    }
    os << "\n";
    return os;
  }



  inline void compute_offsets() {
    offsets[rank-1] = 1;
    for (int i=rank-2; i>=0; i--) {
      offsets[i] = offsets[i+1] * dimension[i+1];
    }
  }


  inline void allocate(char const * label = "") {
    if (owned) {
      static_assert( std::is_arithmetic<T>() || myMem == memHost , 
                     "ERROR: You cannot use non-arithmetic types inside owned Arrays on the device" );
      refCount = new int;
      *refCount = 1;
      if (myMem == memDevice) {
        myData = (T *) yaklAllocDevice( totElems()*sizeof(T) , label );
      } else {
        myData = new T[totElems()];
      }
    }
  }


  YAKL_INLINE void deallocate() {
    if (owned) {
      if (refCount != nullptr) {
        (*refCount)--;

        if (*refCount == 0) {
          delete refCount;
          refCount = nullptr;
          if (myMem == memDevice) {
            #ifdef YAKL_DEBUG
              yaklFreeDevice(myData,myname.c_str());
            #else
              yaklFreeDevice(myData,"");
            #endif
          } else {
            delete[] myData;
          }
          myData = nullptr;
        }

      }
    }
  }

};

