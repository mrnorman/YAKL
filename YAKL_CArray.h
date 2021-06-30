
#pragma once

template <class T, int rank, int myMem> class Array<T,rank,myMem,styleC> {
public:

  T       * myData;         // Pointer to the flattened internal data
  int     * refCount;       // Pointer shared by multiple copies of this Array to keep track of allcation / free
  index_t dimension[rank];  // Sizes of the 8 possible dimensions
  bool    owned;            // Whether is is owned (owned = allocated,ref_counted,deallocated) or not
  #ifdef YAKL_DEBUG
    char const * myname;          // Label for debug printing. Only stored if debugging is turned on
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
      myname = label;
    #endif
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Owned constructors
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Array(char const * label, index_t const d1) {
    #ifdef YAKL_DEBUG
      if( rank != 1 ) { yakl_throw("ERROR: Calling invalid constructor on rank 1 Array"); }
      myname = label;
    #endif
    nullify();
    deallocate();
    dimension[0] = d1;
    allocate(label);
  }
  Array(char const * label, index_t const d1, index_t const d2) {
    #ifdef YAKL_DEBUG
      if( rank != 2 ) { yakl_throw("ERROR: Calling invalid constructor on rank 2 Array"); }
      myname = label;
    #endif
    nullify();
    deallocate();
    dimension[0] = d1;
    dimension[1] = d2;
    allocate(label);
  }
  Array(char const * label, index_t const d1, index_t const d2, index_t const d3) {
    #ifdef YAKL_DEBUG
      if( rank != 3 ) { yakl_throw("ERROR: Calling invalid constructor on rank 3 Array"); }
      myname = label;
    #endif
    nullify();
    deallocate();
    dimension[0] = d1;
    dimension[1] = d2;
    dimension[2] = d3;
    allocate(label);
  }
  Array(char const * label, index_t const d1, index_t const d2, index_t const d3, index_t const d4) {
    #ifdef YAKL_DEBUG
      if( rank != 4 ) { yakl_throw("ERROR: Calling invalid constructor on rank 4 Array"); }
      myname = label;
    #endif
    nullify();
    deallocate();
    dimension[0] = d1;
    dimension[1] = d2;
    dimension[2] = d3;
    dimension[3] = d4;
    allocate(label);
  }
  Array(char const * label, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5) {
    #ifdef YAKL_DEBUG
      if( rank != 5 ) { yakl_throw("ERROR: Calling invalid constructor on rank 5 Array"); }
      myname = label;
    #endif
    nullify();
    deallocate();
    dimension[0] = d1;
    dimension[1] = d2;
    dimension[2] = d3;
    dimension[3] = d4;
    dimension[4] = d5;
    allocate(label);
  }
  Array(char const * label, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5, index_t const d6) {
    #ifdef YAKL_DEBUG
      if( rank != 6 ) { yakl_throw("ERROR: Calling invalid constructor on rank 6 Array"); }
      myname = label;
    #endif
    nullify();
    deallocate();
    dimension[0] = d1;
    dimension[1] = d2;
    dimension[2] = d3;
    dimension[3] = d4;
    dimension[4] = d5;
    dimension[5] = d6;
    allocate(label);
  }
  Array(char const * label, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5, index_t const d6, index_t const d7) {
    #ifdef YAKL_DEBUG
      if( rank != 7 ) { yakl_throw("ERROR: Calling invalid constructor on rank 7 Array"); }
      myname = label;
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
    allocate(label);
  }
  Array(char const * label, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5, index_t const d6, index_t const d7, index_t const d8) {
    #ifdef YAKL_DEBUG
      if( rank != 8 ) { yakl_throw("ERROR: Calling invalid constructor on rank 8 Array"); }
      myname = label;
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
    allocate(label);
  }
  template <class INT, typename std::enable_if< std::is_integral<INT>::value , int >::type = 0>
  Array(char const * label, std::vector<INT> const dims) {
    #ifdef YAKL_DEBUG
      if ( dims.size() < rank ) { yakl_throw("ERROR: dims < rank"); }
      if ( rank < 1 || rank > 8 ) { yakl_throw("ERROR: Invalid rank, must be between 1 and 8"); }
      myname = label;
    #endif
    nullify();
    deallocate();
    for (int i=0; i < rank; i++) {
      dimension[i] = dims[i];
    }
    allocate(label);
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Non-owned constructors
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Array(char const * label, T * data, index_t const d1) {
    #ifdef YAKL_DEBUG
      if( rank != 1 ) { yakl_throw("ERROR: Calling invalid constructor on rank 1 Array"); }
      myname = label;
    #endif
    nullify();
    owned = false;
    dimension[0] = d1;
    myData = data;
  }
  Array(char const * label, T * data, index_t const d1, index_t const d2) {
    #ifdef YAKL_DEBUG
      if( rank != 2 ) { yakl_throw("ERROR: Calling invalid constructor on rank 2 Array"); }
      myname = label;
    #endif
    nullify();
    owned = false;
    dimension[0] = d1;
    dimension[1] = d2;
    myData = data;
  }
  Array(char const * label, T * data, index_t const d1, index_t const d2, index_t const d3) {
    #ifdef YAKL_DEBUG
      if( rank != 3 ) { yakl_throw("ERROR: Calling invalid constructor on rank 3 Array"); }
      myname = label;
    #endif
    nullify();
    owned = false;
    dimension[0] = d1;
    dimension[1] = d2;
    dimension[2] = d3;
    myData = data;
  }
  Array(char const * label, T * data, index_t const d1, index_t const d2, index_t const d3, index_t const d4) {
    #ifdef YAKL_DEBUG
      if( rank != 4 ) { yakl_throw("ERROR: Calling invalid constructor on rank 4 Array"); }
      myname = label;
    #endif
    nullify();
    owned = false;
    dimension[0] = d1;
    dimension[1] = d2;
    dimension[2] = d3;
    dimension[3] = d4;
    myData = data;
  }
  Array(char const * label, T * data, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5) {
    #ifdef YAKL_DEBUG
      if( rank != 5 ) { yakl_throw("ERROR: Calling invalid constructor on rank 5 Array"); }
      myname = label;
    #endif
    nullify();
    owned = false;
    dimension[0] = d1;
    dimension[1] = d2;
    dimension[2] = d3;
    dimension[3] = d4;
    dimension[4] = d5;
    myData = data;
  }
  Array(char const * label, T * data, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5, index_t const d6) {
    #ifdef YAKL_DEBUG
      if( rank != 6 ) { yakl_throw("ERROR: Calling invalid constructor on rank 6 Array"); }
      myname = label;
    #endif
    nullify();
    owned = false;
    dimension[0] = d1;
    dimension[1] = d2;
    dimension[2] = d3;
    dimension[3] = d4;
    dimension[4] = d5;
    dimension[5] = d6;
    myData = data;
  }
  Array(char const * label, T * data, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5, index_t const d6, index_t const d7) {
    #ifdef YAKL_DEBUG
      if( rank != 7 ) { yakl_throw("ERROR: Calling invalid constructor on rank 7 Array"); }
      myname = label;
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
    myData = data;
  }
  Array(char const * label, T * data, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5, index_t const d6, index_t const d7, index_t const d8) {
    #ifdef YAKL_DEBUG
      if( rank != 8 ) { yakl_throw("ERROR: Calling invalid constructor on rank 8 Array"); }
      myname = label;
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
    myData = data;
  }
  template <class INT, typename std::enable_if< std::is_integral<INT>::value , int >::type = 0>
  Array(char const * label, T * data, std::vector<INT> const dims) {
    #ifdef YAKL_DEBUG
      if ( dims.size() < rank ) { yakl_throw("ERROR: dims < rank"); }
      if ( rank < 1 || rank > 8 ) { yakl_throw("ERROR: Invalid rank, must be between 1 and 8"); }
      myname = label;
    #endif
    nullify();
    owned = false;
    for (int i=0; i < rank; i++) {
      dimension[i] = dims[i];
    }
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
      dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      myname = rhs.myname;
    #endif
    myData   = rhs.myData;
    refCount = rhs.refCount;
    if (owned && refCount != nullptr) { (*refCount)++; }
  }


  Array & operator=(Array const &rhs) {
    if (this == &rhs) {
      return *this;
    }
    owned    = rhs.owned;
    deallocate();
    for (int i=0; i<rank; i++) {
      dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      myname = rhs.myname;
    #endif
    myData   = rhs.myData;
    refCount = rhs.refCount;
    if (owned && refCount != nullptr) { (*refCount)++; }

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
      if ( rank != 1 || i0 >= dimension[0] ) { indexing_check(1,i0); };
    #endif
    index_t ind = i0;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(index_t const i0, index_t const i1) const {
    #ifdef YAKL_DEBUG
      if ( rank != 2 || i0 >= dimension[0] ||
                        i1 >= dimension[1] ) { indexing_check(2,i0,i1); };
    #endif
    index_t ind = i0*dimension[1] + i1;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(index_t const i0, index_t const i1, index_t const i2) const {
    #ifdef YAKL_DEBUG
      if ( rank != 3 || i0 >= dimension[0] ||
                        i1 >= dimension[1] ||
                        i2 >= dimension[2] ) { indexing_check(3,i0,i1,i2); };
    #endif
    index_t ind = (i0*dimension[1] + i1)*dimension[2] + i2;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(index_t const i0, index_t const i1, index_t const i2, index_t const i3) const {
    #ifdef YAKL_DEBUG
      if ( rank != 4 || i0 >= dimension[0] ||
                        i1 >= dimension[1] ||
                        i2 >= dimension[2] ||
                        i3 >= dimension[3] ) { indexing_check(4,i0,i1,i2,i3); };
    #endif
    index_t ind = ((i0*dimension[1] + i1)*dimension[2] + i2)*dimension[3] + i3;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(index_t const i0, index_t const i1, index_t const i2, index_t const i3, index_t const i4) const {
    #ifdef YAKL_DEBUG
      if ( rank != 5 || i0 >= dimension[0] ||
                        i1 >= dimension[1] ||
                        i2 >= dimension[2] ||
                        i3 >= dimension[3] ||
                        i4 >= dimension[4] ) { indexing_check(5,i0,i1,i2,i3,i4); };
    #endif
    index_t ind = (((i0*dimension[1] + i1)*dimension[2] + i2)*dimension[3] + i3)*dimension[4] + i4;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(index_t const i0, index_t const i1, index_t const i2, index_t const i3, index_t const i4, index_t const i5) const {
    #ifdef YAKL_DEBUG
      if ( rank != 6 || i0 >= dimension[0] ||
                        i1 >= dimension[1] ||
                        i2 >= dimension[2] ||
                        i3 >= dimension[3] ||
                        i4 >= dimension[4] ||
                        i5 >= dimension[5] ) { indexing_check(6,i0,i1,i2,i3,i4,i5); };
    #endif
    index_t ind = ((((i0*dimension[1] + i1)*dimension[2] + i2)*dimension[3] + i3)*dimension[4] + i4)*dimension[5] + i5;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(index_t const i0, index_t const i1, index_t const i2, index_t const i3, index_t const i4, index_t const i5, index_t const i6) const {
    #ifdef YAKL_DEBUG
      if ( rank != 7 || i0 >= dimension[0] ||
                        i1 >= dimension[1] ||
                        i2 >= dimension[2] ||
                        i3 >= dimension[3] ||
                        i4 >= dimension[4] ||
                        i5 >= dimension[5] ||
                        i6 >= dimension[6] ) { indexing_check(7,i0,i1,i2,i3,i4,i5,i6); };
    #endif
    index_t ind = (((((i0*dimension[1] + i1)*dimension[2] + i2)*dimension[3] + i3)*dimension[4] + i4)*dimension[5] + i5)*dimension[6] + i6;
    return myData[ind];
  }
  YAKL_INLINE T &operator()(index_t const i0, index_t const i1, index_t const i2, index_t const i3, index_t const i4, index_t const i5, index_t const i6, index_t const i7) const {
    #ifdef YAKL_DEBUG
      if ( rank != 8 || i0 >= dimension[0] ||
                        i1 >= dimension[1] ||
                        i2 >= dimension[2] ||
                        i3 >= dimension[3] ||
                        i4 >= dimension[4] ||
                        i5 >= dimension[5] ||
                        i6 >= dimension[6] ||
                        i7 >= dimension[7] ) { indexing_check(8,i0,i1,i2,i3,i4,i5,i6,i7); };
    #endif
    index_t ind = ((((((i0*dimension[1] + i1)*dimension[2] + i2)*dimension[3] + i3)*dimension[4] + i4)*dimension[5] + i5)*dimension[6] + i6)*dimension[7] + i7;
    return myData[ind];
  }


  // if this function gets called, then there was definitely an error
  inline void indexing_check(int rank_in, index_t i0 ,
                                          index_t i1=INDEX_MAX ,
                                          index_t i2=INDEX_MAX ,
                                          index_t i3=INDEX_MAX ,
                                          index_t i4=INDEX_MAX ,
                                          index_t i5=INDEX_MAX ,
                                          index_t i6=INDEX_MAX ,
                                          index_t i7=INDEX_MAX ) const {
    #ifdef YAKL_DEBUG
      #ifndef YAKL_SEPARATE_MEMORY_SPACE
        std::cerr << "For Array labeled: " << myname << ":" << std::endl;
        if (rank_in != rank) { std::cerr << "Indexing with the incorrect number of dimensions. " << std::endl; }
        if (rank >= 1 && i0 >= dimension[0]) { std::cerr << "Index 1 of " << rank << " is out of bounds. Value: " << i0 << "; Bound: " << dimension[0]-1 << std::endl; }
        if (rank >= 2 && i1 >= dimension[1]) { std::cerr << "Index 2 of " << rank << " is out of bounds. Value: " << i1 << "; Bound: " << dimension[1]-1 << std::endl; }
        if (rank >= 3 && i2 >= dimension[2]) { std::cerr << "Index 3 of " << rank << " is out of bounds. Value: " << i2 << "; Bound: " << dimension[2]-1 << std::endl; }
        if (rank >= 4 && i3 >= dimension[3]) { std::cerr << "Index 4 of " << rank << " is out of bounds. Value: " << i3 << "; Bound: " << dimension[3]-1 << std::endl; }
        if (rank >= 5 && i4 >= dimension[4]) { std::cerr << "Index 5 of " << rank << " is out of bounds. Value: " << i4 << "; Bound: " << dimension[4]-1 << std::endl; }
        if (rank >= 6 && i5 >= dimension[5]) { std::cerr << "Index 6 of " << rank << " is out of bounds. Value: " << i5 << "; Bound: " << dimension[5]-1 << std::endl; }
        if (rank >= 7 && i6 >= dimension[6]) { std::cerr << "Index 7 of " << rank << " is out of bounds. Value: " << i6 << "; Bound: " << dimension[6]-1 << std::endl; }
        if (rank >= 8 && i7 >= dimension[7]) { std::cerr << "Index 8 of " << rank << " is out of bounds. Value: " << i7 << "; Bound: " << dimension[7]-1 << std::endl; }
      #endif
      yakl_throw("Invalid Array Index Encountered");
    #endif
  }


  template <int N> YAKL_INLINE void slice( Dims const &dims , Array<T,N,myMem,styleC> &store ) const {
    #ifdef YAKL_DEBUG
      if (rank != dims.size()) {
        yakl_throw( "ERROR: slice rank must be equal to dims.size()" );
      }
      for (int i = rank-1-N; i >= 0; i--) {
        if (dims.data[i] >= dimension[i]) {
          yakl_throw( "ERROR: One of the slicing dimension dimensions is out of bounds" );
        }
      }
    #endif
    store.owned = false;
    index_t offset = 1;
    for (int i = rank-1; i > rank-1-N; i--) {
      store.dimension[i-(rank-N)] = dimension[i];
      offset *= dimension[i];
    }
    index_t retOff = 0;
    for (int i = rank-1-N; i >= 0; i--) {
      retOff += dims.data[i]*offset;
      offset *= dimension[i];
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
      ret.dimension[i] = dimension[i];
    }
    #ifdef YAKL_DEBUG
      ret.myname = myname;
    #endif
    ret.allocate();
    if (myMem == memHost) {
      memcpy_host_to_host( ret.myData , myData , totElems() );
    } else {
      memcpy_device_to_host( ret.myData , myData , totElems() );
    }
    fence();
    return ret;
  }


  inline Array<T,rank,memDevice,styleC> createDeviceCopy() const {
    Array<T,rank,memDevice,styleC> ret;  // nullified + owned == true
    for (int i=0; i<rank; i++) {
      ret.dimension[i] = dimension[i];
    }
    #ifdef YAKL_DEBUG
      ret.myname = myname;
    #endif
    ret.allocate();
    if (myMem == memHost) {
      memcpy_host_to_device( ret.myData , myData , totElems() );
    } else {
      memcpy_device_to_device( ret.myData , myData , totElems() );
    }
    fence();
    return ret;
  }


  template <int N> YAKL_INLINE Array<T,N,myMem,styleC> reshape(Dims const &dims) const {
    #ifdef YAKL_DEBUG
      if (dims.size() != N) { yakl_throw("ERROR: new number of reshaped array dimensions does not match the templated rank"); }
      index_t totelems = 1;
      for (int i=0; i < N; i++) {
        totelems *= dims.data[i];
      }
      if (totelems != this->totElems()) { yakl_throw("ERROR: Total reshaped array elements is not consistent with this array"); }
    #endif
    Array<T,N,myMem,styleC> ret;
    ret.owned = owned;
    for (int i=0; i < N; i++) {
      ret.dimension[i] = dims.data[i];
    }
    #ifdef YAKL_DEBUG
      ret.myname = myname;
    #endif
    ret.myData = myData;
    ret.refCount = refCount;
    if (owned && refCount != nullptr) { (*refCount)++; }
    return ret;
  }


  YAKL_INLINE Array<T,1,myMem,styleC> collapse() {
    Array<T,1,myMem,styleC> ret;
    ret.owned = owned;
    ret.dimension[0] = totElems();
    #ifdef YAKL_DEBUG
      ret.myname = myname;
    #endif
    ret.myData = myData;
    ret.refCount = refCount;
    if (owned && refCount != nullptr) { (*refCount)++; }
    return ret;
  }


  inline void deep_copy_to(Array<T,rank,memHost,styleC> lhs) {
    if (myMem == memHost) {
      memcpy_host_to_host( lhs.myData , myData , totElems() );
    } else {
      memcpy_device_to_host( lhs.myData , myData , totElems() );
    }
  }


  inline void deep_copy_to(Array<T,rank,memDevice,styleC> lhs) {
    if (myMem == memHost) {
      memcpy_host_to_device( lhs.myData , myData , totElems() );
    } else {
      memcpy_device_to_device( lhs.myData , myData , totElems() );
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
      return myname;
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
          if (totElems() > 0) {
            if (myMem == memDevice) {
              #ifdef YAKL_DEBUG
                yaklFreeDevice(myData,myname);
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
  }

};

