
#pragma once

template <class T, int rank, int myMem>
class Array<T,rank,myMem,styleC> : public CArrayBase<T,rank,myMem> {
public:

  int     * refCount;       // Pointer shared by multiple copies of this Array to keep track of allcation / free


  // Start off all constructors making sure the pointers are null
  YAKL_INLINE void nullify() {
    this->myData   = nullptr;
    this->refCount = nullptr;
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
      this->myname = label;
    #endif
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Owned constructors
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  YAKL_INLINE Array(char const * label, index_t const d1) {
    #ifdef YAKL_DEBUG
      if( rank != 1 ) { yakl_throw("ERROR: Calling invalid constructor on rank 1 Array"); }
      this->myname = label;
    #endif
    nullify();
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
    this->dimension[0] = d1;
    #if YAKL_CURRENTLY_ON_HOST()
      allocate(label);
    #endif
  }
  YAKL_INLINE Array(char const * label, index_t const d1, index_t const d2) {
    #ifdef YAKL_DEBUG
      if( rank != 2 ) { yakl_throw("ERROR: Calling invalid constructor on rank 2 Array"); }
      this->myname = label;
    #endif
    nullify();
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
    this->dimension[0] = d1;
    this->dimension[1] = d2;
    #if YAKL_CURRENTLY_ON_HOST()
      allocate(label);
    #endif
  }
  YAKL_INLINE Array(char const * label, index_t const d1, index_t const d2, index_t const d3) {
    #ifdef YAKL_DEBUG
      if( rank != 3 ) { yakl_throw("ERROR: Calling invalid constructor on rank 3 Array"); }
      this->myname = label;
    #endif
    nullify();
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
    this->dimension[0] = d1;
    this->dimension[1] = d2;
    this->dimension[2] = d3;
    #if YAKL_CURRENTLY_ON_HOST()
      allocate(label);
    #endif
  }
  YAKL_INLINE Array(char const * label, index_t const d1, index_t const d2, index_t const d3, index_t const d4) {
    #ifdef YAKL_DEBUG
      if( rank != 4 ) { yakl_throw("ERROR: Calling invalid constructor on rank 4 Array"); }
      this->myname = label;
    #endif
    nullify();
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
    this->dimension[0] = d1;
    this->dimension[1] = d2;
    this->dimension[2] = d3;
    this->dimension[3] = d4;
    #if YAKL_CURRENTLY_ON_HOST()
      allocate(label);
    #endif
  }
  YAKL_INLINE Array(char const * label, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5) {
    #ifdef YAKL_DEBUG
      if( rank != 5 ) { yakl_throw("ERROR: Calling invalid constructor on rank 5 Array"); }
      this->myname = label;
    #endif
    nullify();
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
    this->dimension[0] = d1;
    this->dimension[1] = d2;
    this->dimension[2] = d3;
    this->dimension[3] = d4;
    this->dimension[4] = d5;
    #if YAKL_CURRENTLY_ON_HOST()
      allocate(label);
    #endif
  }
  YAKL_INLINE Array(char const * label, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5, index_t const d6) {
    #ifdef YAKL_DEBUG
      if( rank != 6 ) { yakl_throw("ERROR: Calling invalid constructor on rank 6 Array"); }
      this->myname = label;
    #endif
    nullify();
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
    this->dimension[0] = d1;
    this->dimension[1] = d2;
    this->dimension[2] = d3;
    this->dimension[3] = d4;
    this->dimension[4] = d5;
    this->dimension[5] = d6;
    #if YAKL_CURRENTLY_ON_HOST()
      allocate(label);
    #endif
  }
  YAKL_INLINE Array(char const * label, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5, index_t const d6, index_t const d7) {
    #ifdef YAKL_DEBUG
      if( rank != 7 ) { yakl_throw("ERROR: Calling invalid constructor on rank 7 Array"); }
      this->myname = label;
    #endif
    nullify();
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
    this->dimension[0] = d1;
    this->dimension[1] = d2;
    this->dimension[2] = d3;
    this->dimension[3] = d4;
    this->dimension[4] = d5;
    this->dimension[5] = d6;
    this->dimension[6] = d7;
    #if YAKL_CURRENTLY_ON_HOST()
      allocate(label);
    #endif
  }
  YAKL_INLINE Array(char const * label, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5, index_t const d6, index_t const d7, index_t const d8) {
    #ifdef YAKL_DEBUG
      if( rank != 8 ) { yakl_throw("ERROR: Calling invalid constructor on rank 8 Array"); }
      this->myname = label;
    #endif
    nullify();
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
    this->dimension[0] = d1;
    this->dimension[1] = d2;
    this->dimension[2] = d3;
    this->dimension[3] = d4;
    this->dimension[4] = d5;
    this->dimension[5] = d6;
    this->dimension[6] = d7;
    this->dimension[7] = d8;
    #if YAKL_CURRENTLY_ON_HOST()
      allocate(label);
    #endif
  }
  template <class INT, typename std::enable_if< std::is_integral<INT>::value , int >::type = 0>
  Array(char const * label, std::vector<INT> const dims) {
    #ifdef YAKL_DEBUG
      if ( dims.size() < rank ) { yakl_throw("ERROR: dims < rank"); }
      if ( rank < 1 || rank > 8 ) { yakl_throw("ERROR: Invalid rank, must be between 1 and 8"); }
      this->myname = label;
    #endif
    nullify();
    deallocate();
    for (int i=0; i < rank; i++) {
      this->dimension[i] = dims[i];
    }
    allocate(label);
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Non-owned constructors
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  YAKL_INLINE Array(char const * label, T * data, index_t const d1) {
    #ifdef YAKL_DEBUG
      if( rank != 1 ) { yakl_throw("ERROR: Calling invalid constructor on rank 1 Array"); }
      this->myname = label;
    #endif
    nullify();
    this->dimension[0] = d1;
    this->myData = data;
    this->refCount = nullptr;
  }
  YAKL_INLINE Array(char const * label, T * data, index_t const d1, index_t const d2) {
    #ifdef YAKL_DEBUG
      if( rank != 2 ) { yakl_throw("ERROR: Calling invalid constructor on rank 2 Array"); }
      this->myname = label;
    #endif
    nullify();
    this->dimension[0] = d1;
    this->dimension[1] = d2;
    this->myData = data;
    this->refCount = nullptr;
  }
  YAKL_INLINE Array(char const * label, T * data, index_t const d1, index_t const d2, index_t const d3) {
    #ifdef YAKL_DEBUG
      if( rank != 3 ) { yakl_throw("ERROR: Calling invalid constructor on rank 3 Array"); }
      this->myname = label;
    #endif
    nullify();
    this->dimension[0] = d1;
    this->dimension[1] = d2;
    this->dimension[2] = d3;
    this->myData = data;
    this->refCount = nullptr;
  }
  YAKL_INLINE Array(char const * label, T * data, index_t const d1, index_t const d2, index_t const d3, index_t const d4) {
    #ifdef YAKL_DEBUG
      if( rank != 4 ) { yakl_throw("ERROR: Calling invalid constructor on rank 4 Array"); }
      this->myname = label;
    #endif
    nullify();
    this->dimension[0] = d1;
    this->dimension[1] = d2;
    this->dimension[2] = d3;
    this->dimension[3] = d4;
    this->myData = data;
    this->refCount = nullptr;
  }
  YAKL_INLINE Array(char const * label, T * data, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5) {
    #ifdef YAKL_DEBUG
      if( rank != 5 ) { yakl_throw("ERROR: Calling invalid constructor on rank 5 Array"); }
      this->myname = label;
    #endif
    nullify();
    this->dimension[0] = d1;
    this->dimension[1] = d2;
    this->dimension[2] = d3;
    this->dimension[3] = d4;
    this->dimension[4] = d5;
    this->myData = data;
    this->refCount = nullptr;
  }
  YAKL_INLINE Array(char const * label, T * data, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5, index_t const d6) {
    #ifdef YAKL_DEBUG
      if( rank != 6 ) { yakl_throw("ERROR: Calling invalid constructor on rank 6 Array"); }
      this->myname = label;
    #endif
    nullify();
    this->dimension[0] = d1;
    this->dimension[1] = d2;
    this->dimension[2] = d3;
    this->dimension[3] = d4;
    this->dimension[4] = d5;
    this->dimension[5] = d6;
    this->myData = data;
    this->refCount = nullptr;
  }
  YAKL_INLINE Array(char const * label, T * data, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5, index_t const d6, index_t const d7) {
    #ifdef YAKL_DEBUG
      if( rank != 7 ) { yakl_throw("ERROR: Calling invalid constructor on rank 7 Array"); }
      this->myname = label;
    #endif
    nullify();
    this->dimension[0] = d1;
    this->dimension[1] = d2;
    this->dimension[2] = d3;
    this->dimension[3] = d4;
    this->dimension[4] = d5;
    this->dimension[5] = d6;
    this->dimension[6] = d7;
    this->myData = data;
    this->refCount = nullptr;
  }
  YAKL_INLINE Array(char const * label, T * data, index_t const d1, index_t const d2, index_t const d3, index_t const d4, index_t const d5, index_t const d6, index_t const d7, index_t const d8) {
    #ifdef YAKL_DEBUG
      if( rank != 8 ) { yakl_throw("ERROR: Calling invalid constructor on rank 8 Array"); }
      this->myname = label;
    #endif
    nullify();
    this->dimension[0] = d1;
    this->dimension[1] = d2;
    this->dimension[2] = d3;
    this->dimension[3] = d4;
    this->dimension[4] = d5;
    this->dimension[5] = d6;
    this->dimension[6] = d7;
    this->dimension[7] = d8;
    this->myData = data;
    this->refCount = nullptr;
  }
  template <class INT, typename std::enable_if< std::is_integral<INT>::value , int >::type = 0>
  inline Array(char const * label, T * data, std::vector<INT> const dims) {
    #ifdef YAKL_DEBUG
      if ( dims.size() < rank ) { yakl_throw("ERROR: dims < rank"); }
      if ( rank < 1 || rank > 8 ) { yakl_throw("ERROR: Invalid rank, must be between 1 and 8"); }
      this->myname = label;
    #endif
    nullify();
    for (int i=0; i < rank; i++) {
      this->dimension[i] = dims[i];
    }
    this->myData = data;
    this->refCount = nullptr;
  }


  /*
  COPY CONSTRUCTORS / FUNCTIONS
  This shares the pointers with another Array and increments the refCounter
  */
  YAKL_INLINE Array(Array const &rhs) {
    // constructor, so no need to deallocate
    nullify();
    for (int i=0; i<rank; i++) {
      this->dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      this->myname = rhs.myname;
    #endif
    this->myData   = rhs.myData;
    this->refCount = rhs.refCount;
    if (this->refCount != nullptr) {
      #if YAKL_CURRENTLY_ON_HOST()
        yakl_mtx_lock();
        (*this->refCount)++;
        yakl_mtx_unlock();
      #endif
    }
  }


  YAKL_INLINE Array & operator=(Array const &rhs) {
    if (this == &rhs) {
      return *this;
    }
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
    for (int i=0; i<rank; i++) {
      this->dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      this->myname = rhs.myname;
    #endif
    this->myData   = rhs.myData;
    this->refCount = rhs.refCount;
    if (this->refCount != nullptr) {
      #if YAKL_CURRENTLY_ON_HOST()
        yakl_mtx_lock();
        (*this->refCount)++;
        yakl_mtx_unlock();
      #endif
    }

    return *this;
  }


  /*
  MOVE CONSTRUCTORS
  This steals the pointers form the rhs rather than sharing and sets rhs pointers to nullptr.
  Therefore, no need to increment refCout
  */
  YAKL_INLINE Array(Array &&rhs) {
    // constructor, so no need to deallocate
    nullify();
    for (int i=0; i<rank; i++) {
      this->dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      this->myname = rhs.myname;
    #endif
    this->myData   = rhs.myData;
    this->refCount = rhs.refCount;

    rhs.myData   = nullptr;
    rhs.refCount = nullptr;
  }


  YAKL_INLINE Array& operator=(Array &&rhs) {
    if (this == &rhs) { return *this; }
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
    for (int i=0; i<rank; i++) {
      this->dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      this->myname = rhs.myname;
    #endif
    this->myData   = rhs.myData;
    this->refCount = rhs.refCount;

    rhs.myData   = nullptr;
    rhs.refCount = nullptr;

    return *this;
  }


  /*
  DESTRUCTOR
  Decrement the refCounter, and if it's zero, deallocate and nullify.  
  */
  YAKL_INLINE ~Array() {
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
  }


  template <int N> inline Array<T,N,myMem,styleC> reshape(Dims const &dims) const {
    #ifdef YAKL_DEBUG
      if (dims.size() != N) { yakl_throw("ERROR: new number of reshaped array dimensions does not match the templated rank"); }
      index_t totelems = 1;
      for (int i=0; i < N; i++) {
        totelems *= dims.data[i];
      }
      if (totelems != this->totElems()) { yakl_throw("ERROR: Total reshaped array elements is not consistent with this array"); }
    #endif
    Array<T,N,myMem,styleC> ret;
    for (int i=0; i < N; i++) {
      ret.dimension[i] = dims.data[i];
    }
    #ifdef YAKL_DEBUG
      ret.myname = this->myname;
    #endif
    ret.myData = this->myData;
    ret.refCount = this->refCount;
    if (this->refCount != nullptr) {
      #if YAKL_CURRENTLY_ON_HOST()
        yakl_mtx_lock();
        (*this->refCount)++;
        yakl_mtx_unlock();
      #endif
    }
    return ret;
  }


  inline Array<T,1,myMem,styleC> collapse() const {
    Array<T,1,myMem,styleC> ret;
    ret.dimension[0] = this->totElems();
    #ifdef YAKL_DEBUG
      ret.myname = this->myname;
    #endif
    ret.myData = this->myData;
    ret.refCount = this->refCount;
    if (this->refCount != nullptr) {
      #if YAKL_CURRENTLY_ON_HOST()
        yakl_mtx_lock();
        (*this->refCount)++;
        yakl_mtx_unlock();
      #endif
    }
    return ret;
  }


  /* ACCESSORS */
  inline int use_count() const {
    return *this->refCount;
  }


  inline void allocate(char const * label = "") {
    // static_assert( std::is_arithmetic<T>() || myMem == memHost , 
    //                "ERROR: You cannot use non-arithmetic types inside owned Arrays on the device" );
    this->refCount = new int;
    *this->refCount = 1;
    if (myMem == memDevice) {
      this->myData = (T *) yaklAllocDevice( this->totElems()*sizeof(T) , label );
    } else {
      this->myData = new T[this->totElems()];
    }
  }


  inline void deallocate() {
    if (this->refCount != nullptr) {
      yakl_mtx_lock();
      (*this->refCount)--;
      yakl_mtx_unlock();

      if (*this->refCount == 0) {
        delete this->refCount;
        this->refCount = nullptr;
        if (this->totElems() > 0) {
          if (myMem == memDevice) {
            #ifdef YAKL_DEBUG
              yaklFreeDevice(this->myData,this->myname);
            #else
              yaklFreeDevice(this->myData,"");
            #endif
          } else {
            delete[] this->myData;
          }
          this->myData = nullptr;
        }
      }

    }
  }

};



