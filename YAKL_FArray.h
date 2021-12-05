
#pragma once

template <class T, int rank, int myMem>
class Array<T,rank,myMem,styleFortran> : public FArrayBase<T,rank,myMem> {
public:

  typedef typename std::remove_cv<T>::type type;
  typedef          T value_type;
  typedef typename std::add_const<type>::type const_value_type;
  typedef typename std::remove_const<type>::type non_const_value_type;

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
  YAKL_INLINE Array(char const * label, Bnd const &b1) {
    #ifdef YAKL_DEBUG
      if ( rank != 1 ) { yakl_throw("ERROR: Calling invalid constructor on rank 1 Array"); }
    #endif
    nullify();
    #ifdef YAKL_DEBUG
      this->myname = label;
    #endif
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
    this->lbounds[0] = b1.l; this->dimension[0] = b1.u - b1.l + 1;  
    #if YAKL_CURRENTLY_ON_HOST()
      allocate(label);
    #endif
  }
  YAKL_INLINE Array(char const * label, Bnd const &b1, Bnd const &b2) {
    #ifdef YAKL_DEBUG
      if ( rank != 2 ) { yakl_throw("ERROR: Calling invalid constructor on rank 2 Array"); }
    #endif
    nullify();
    #ifdef YAKL_DEBUG
      this->myname = label;
    #endif
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
    this->lbounds[0] = b1.l; this->dimension[0] = b1.u - b1.l + 1;
    this->lbounds[1] = b2.l; this->dimension[1] = b2.u - b2.l + 1;
    #if YAKL_CURRENTLY_ON_HOST()
      allocate(label);
    #endif
  }
  YAKL_INLINE Array(char const * label, Bnd const &b1, Bnd const &b2, Bnd const &b3) {
    #ifdef YAKL_DEBUG
      if ( rank != 3 ) { yakl_throw("ERROR: Calling invalid constructor on rank 3 Array"); }
    #endif
    nullify();
    #ifdef YAKL_DEBUG
      this->myname = label;
    #endif
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
    this->lbounds[0] = b1.l; this->dimension[0] = b1.u - b1.l + 1;
    this->lbounds[1] = b2.l; this->dimension[1] = b2.u - b2.l + 1;
    this->lbounds[2] = b3.l; this->dimension[2] = b3.u - b3.l + 1;
    #if YAKL_CURRENTLY_ON_HOST()
      allocate(label);
    #endif
  }
  YAKL_INLINE Array(char const * label, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4) {
    #ifdef YAKL_DEBUG
      if ( rank != 4 ) { yakl_throw("ERROR: Calling invalid constructor on rank 4 Array"); }
    #endif
    nullify();
    #ifdef YAKL_DEBUG
      this->myname = label;
    #endif
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
    this->lbounds[0] = b1.l; this->dimension[0] = b1.u - b1.l + 1;
    this->lbounds[1] = b2.l; this->dimension[1] = b2.u - b2.l + 1;
    this->lbounds[2] = b3.l; this->dimension[2] = b3.u - b3.l + 1;
    this->lbounds[3] = b4.l; this->dimension[3] = b4.u - b4.l + 1;
    #if YAKL_CURRENTLY_ON_HOST()
      allocate(label);
    #endif
  }
  YAKL_INLINE Array(char const * label, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5) {
    #ifdef YAKL_DEBUG
      if ( rank != 5 ) { yakl_throw("ERROR: Calling invalid constructor on rank 5 Array"); }
    #endif
    nullify();
    #ifdef YAKL_DEBUG
      this->myname = label;
    #endif
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
    this->lbounds[0] = b1.l; this->dimension[0] = b1.u - b1.l + 1;
    this->lbounds[1] = b2.l; this->dimension[1] = b2.u - b2.l + 1;
    this->lbounds[2] = b3.l; this->dimension[2] = b3.u - b3.l + 1;
    this->lbounds[3] = b4.l; this->dimension[3] = b4.u - b4.l + 1;
    this->lbounds[4] = b5.l; this->dimension[4] = b5.u - b5.l + 1;
    #if YAKL_CURRENTLY_ON_HOST()
      allocate(label);
    #endif
  }
  YAKL_INLINE Array(char const * label, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5, Bnd const &b6) {
    #ifdef YAKL_DEBUG
      if ( rank != 6 ) { yakl_throw("ERROR: Calling invalid constructor on rank 6 Array"); }
    #endif
    nullify();
    #ifdef YAKL_DEBUG
      this->myname = label;
    #endif
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
    this->lbounds[0] = b1.l; this->dimension[0] = b1.u - b1.l + 1;
    this->lbounds[1] = b2.l; this->dimension[1] = b2.u - b2.l + 1;
    this->lbounds[2] = b3.l; this->dimension[2] = b3.u - b3.l + 1;
    this->lbounds[3] = b4.l; this->dimension[3] = b4.u - b4.l + 1;
    this->lbounds[4] = b5.l; this->dimension[4] = b5.u - b5.l + 1;
    this->lbounds[5] = b6.l; this->dimension[5] = b6.u - b6.l + 1;
    #if YAKL_CURRENTLY_ON_HOST()
      allocate(label);
    #endif
  }
  YAKL_INLINE Array(char const * label, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5, Bnd const &b6, Bnd const &b7) {
    #ifdef YAKL_DEBUG
      if ( rank != 7 ) { yakl_throw("ERROR: Calling invalid constructor on rank 7 Array"); }
    #endif
    nullify();
    #ifdef YAKL_DEBUG
      this->myname = label;
    #endif
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
    this->lbounds[0] = b1.l; this->dimension[0] = b1.u - b1.l + 1;
    this->lbounds[1] = b2.l; this->dimension[1] = b2.u - b2.l + 1;
    this->lbounds[2] = b3.l; this->dimension[2] = b3.u - b3.l + 1;
    this->lbounds[3] = b4.l; this->dimension[3] = b4.u - b4.l + 1;
    this->lbounds[4] = b5.l; this->dimension[4] = b5.u - b5.l + 1;
    this->lbounds[5] = b6.l; this->dimension[5] = b6.u - b6.l + 1;
    this->lbounds[6] = b7.l; this->dimension[6] = b7.u - b7.l + 1;
    #if YAKL_CURRENTLY_ON_HOST()
      allocate(label);
    #endif
  }
  YAKL_INLINE Array(char const * label, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5, Bnd const &b6, Bnd const &b7, Bnd const &b8) {
    #ifdef YAKL_DEBUG
      if ( rank != 8 ) { yakl_throw("ERROR: Calling invalid constructor on rank 8 Array"); }
    #endif
    nullify();
    #ifdef YAKL_DEBUG
      this->myname = label;
    #endif
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
    this->lbounds[0] = b1.l; this->dimension[0] = b1.u - b1.l + 1;
    this->lbounds[1] = b2.l; this->dimension[1] = b2.u - b2.l + 1;
    this->lbounds[2] = b3.l; this->dimension[2] = b3.u - b3.l + 1;
    this->lbounds[3] = b4.l; this->dimension[3] = b4.u - b4.l + 1;
    this->lbounds[4] = b5.l; this->dimension[4] = b5.u - b5.l + 1;
    this->lbounds[5] = b6.l; this->dimension[5] = b6.u - b6.l + 1;
    this->lbounds[6] = b7.l; this->dimension[6] = b7.u - b7.l + 1;
    this->lbounds[7] = b8.l; this->dimension[7] = b8.u - b8.l + 1;
    #if YAKL_CURRENTLY_ON_HOST()
      allocate(label);
    #endif
  }
  Array(char const * label, std::vector<Bnd> const &bnds) {
    #ifdef YAKL_DEBUG
      if ( bnds.size() < rank ) { yakl_throw("ERROR: Number of array bounds specified is < rank"); }
      if ( rank < 1 || rank > 8) { yakl_throw("ERROR: Invalid number of ranks. Must be between 1 and 8"); }
    #endif
    nullify();
    #ifdef YAKL_DEBUG
      this->myname = label;
    #endif
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
    for (int i=0; i < rank; i++) {
      this->lbounds[i] = bnds[i].l;
      this->dimension[i] = bnds[i].u - bnds[i].l + 1;
    }
    #if YAKL_CURRENTLY_ON_HOST()
      allocate(label);
    #endif
  }
  template <class INT, typename std::enable_if< std::is_integral<INT>::value , int >::type = 0>
  Array(char const * label, std::vector<INT> const &bnds) {
    #ifdef YAKL_DEBUG
      if ( bnds.size() < rank ) { yakl_throw("ERROR: Number of array bounds specified is < rank"); }
      if ( rank < 1 || rank > 8) { yakl_throw("ERROR: Invalid number of ranks. Must be between 1 and 8"); }
    #endif
    nullify();
    #ifdef YAKL_DEBUG
      this->myname = label;
    #endif
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
    for (int i=0; i < rank; i++) {
      this->lbounds[i] = 1;
      this->dimension[i] = bnds[i];
    }
    #if YAKL_CURRENTLY_ON_HOST()
      allocate(label);
    #endif
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Non-owned constructors
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  YAKL_INLINE Array(char const * label, T * data, Bnd const &b1) {
    #ifdef YAKL_DEBUG
      if ( rank != 1 ) { yakl_throw("ERROR: Calling invalid constructor on rank 1 Array"); }
    #endif
    nullify();
    #ifdef YAKL_DEBUG
      this->myname = label;
    #endif
    this->lbounds[0] = b1.l; this->dimension[0] = b1.u - b1.l + 1;  
    this->myData = data;
    this->refCount = nullptr;
  }
  YAKL_INLINE Array(char const * label, T * data, Bnd const &b1, Bnd const &b2) {
    #ifdef YAKL_DEBUG
      if ( rank != 2 ) { yakl_throw("ERROR: Calling invalid constructor on rank 2 Array"); }
    #endif
    nullify();
    #ifdef YAKL_DEBUG
      this->myname = label;
    #endif
    this->lbounds[0] = b1.l; this->dimension[0] = b1.u - b1.l + 1;  
    this->lbounds[1] = b2.l; this->dimension[1] = b2.u - b2.l + 1;
    this->myData = data;
    this->refCount = nullptr;
  }
  YAKL_INLINE Array(char const * label, T * data, Bnd const &b1, Bnd const &b2, Bnd const &b3) {
    #ifdef YAKL_DEBUG
      if ( rank != 3 ) { yakl_throw("ERROR: Calling invalid constructor on rank 3 Array"); }
    #endif
    nullify();
    #ifdef YAKL_DEBUG
      this->myname = label;
    #endif
    this->lbounds[0] = b1.l; this->dimension[0] = b1.u - b1.l + 1;
    this->lbounds[1] = b2.l; this->dimension[1] = b2.u - b2.l + 1;
    this->lbounds[2] = b3.l; this->dimension[2] = b3.u - b3.l + 1;
    this->myData = data;
    this->refCount = nullptr;
  }
  YAKL_INLINE Array(char const * label, T * data, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4) {
    #ifdef YAKL_DEBUG
      if ( rank != 4 ) { yakl_throw("ERROR: Calling invalid constructor on rank 4 Array"); }
    #endif
    nullify();
    #ifdef YAKL_DEBUG
      this->myname = label;
    #endif
    this->lbounds[0] = b1.l; this->dimension[0] = b1.u - b1.l + 1;
    this->lbounds[1] = b2.l; this->dimension[1] = b2.u - b2.l + 1;
    this->lbounds[2] = b3.l; this->dimension[2] = b3.u - b3.l + 1;
    this->lbounds[3] = b4.l; this->dimension[3] = b4.u - b4.l + 1;
    this->myData = data;
    this->refCount = nullptr;
  }
  YAKL_INLINE Array(char const * label, T * data, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5) {
    #ifdef YAKL_DEBUG
      if ( rank != 5 ) { yakl_throw("ERROR: Calling invalid constructor on rank 5 Array"); }
    #endif
    nullify();
    #ifdef YAKL_DEBUG
      this->myname = label;
    #endif
    this->lbounds[0] = b1.l; this->dimension[0] = b1.u - b1.l + 1;
    this->lbounds[1] = b2.l; this->dimension[1] = b2.u - b2.l + 1;
    this->lbounds[2] = b3.l; this->dimension[2] = b3.u - b3.l + 1;
    this->lbounds[3] = b4.l; this->dimension[3] = b4.u - b4.l + 1;
    this->lbounds[4] = b5.l; this->dimension[4] = b5.u - b5.l + 1;
    this->myData = data;
    this->refCount = nullptr;
  }
  YAKL_INLINE Array(char const * label, T * data, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5, Bnd const &b6) {
    #ifdef YAKL_DEBUG
      if ( rank != 6 ) { yakl_throw("ERROR: Calling invalid constructor on rank 6 Array"); }
    #endif
    nullify();
    #ifdef YAKL_DEBUG
      this->myname = label;
    #endif
    this->lbounds[0] = b1.l; this->dimension[0] = b1.u - b1.l + 1;  
    this->lbounds[1] = b2.l; this->dimension[1] = b2.u - b2.l + 1;
    this->lbounds[2] = b3.l; this->dimension[2] = b3.u - b3.l + 1;
    this->lbounds[3] = b4.l; this->dimension[3] = b4.u - b4.l + 1;
    this->lbounds[4] = b5.l; this->dimension[4] = b5.u - b5.l + 1;
    this->lbounds[5] = b6.l; this->dimension[5] = b6.u - b6.l + 1;
    this->myData = data;
    this->refCount = nullptr;
  }
  YAKL_INLINE Array(char const * label, T * data, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5, Bnd const &b6, Bnd const &b7) {
    #ifdef YAKL_DEBUG
      if ( rank != 7 ) { yakl_throw("ERROR: Calling invalid constructor on rank 7 Array"); }
    #endif
    nullify();
    #ifdef YAKL_DEBUG
      this->myname = label;
    #endif
    this->lbounds[0] = b1.l; this->dimension[0] = b1.u - b1.l + 1;
    this->lbounds[1] = b2.l; this->dimension[1] = b2.u - b2.l + 1;
    this->lbounds[2] = b3.l; this->dimension[2] = b3.u - b3.l + 1;
    this->lbounds[3] = b4.l; this->dimension[3] = b4.u - b4.l + 1;
    this->lbounds[4] = b5.l; this->dimension[4] = b5.u - b5.l + 1;
    this->lbounds[5] = b6.l; this->dimension[5] = b6.u - b6.l + 1;
    this->lbounds[6] = b7.l; this->dimension[6] = b7.u - b7.l + 1;
    this->myData = data;
    this->refCount = nullptr;
  }
  YAKL_INLINE Array(char const * label, T * data, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5, Bnd const &b6, Bnd const &b7, Bnd const &b8) {
    #ifdef YAKL_DEBUG
      if ( rank != 8 ) { yakl_throw("ERROR: Calling invalid constructor on rank 8 Array"); }
    #endif
    nullify();
    #ifdef YAKL_DEBUG
      this->myname = label;
    #endif
    this->lbounds[0] = b1.l; this->dimension[0] = b1.u - b1.l + 1;
    this->lbounds[1] = b2.l; this->dimension[1] = b2.u - b2.l + 1;
    this->lbounds[2] = b3.l; this->dimension[2] = b3.u - b3.l + 1;
    this->lbounds[3] = b4.l; this->dimension[3] = b4.u - b4.l + 1;
    this->lbounds[4] = b5.l; this->dimension[4] = b5.u - b5.l + 1;
    this->lbounds[5] = b6.l; this->dimension[5] = b6.u - b6.l + 1;
    this->lbounds[6] = b7.l; this->dimension[6] = b7.u - b7.l + 1;
    this->lbounds[7] = b8.l; this->dimension[7] = b8.u - b8.l + 1;
    this->myData = data;
    this->refCount = nullptr;
  }
  YAKL_INLINE Array(char const * label, T * data, std::vector<Bnd> const &bnds) {
    #ifdef YAKL_DEBUG
      if ( bnds.size() < rank ) { yakl_throw("ERROR: Number of array bounds specified is < rank"); }
      if ( rank < 1 || rank > 8) { yakl_throw("ERROR: Invalid number of ranks. Must be between 1 and 8"); }
    #endif
    nullify();
    #ifdef YAKL_DEBUG
      this->myname = label;
    #endif
    for (int i=0; i < rank; i++) {
      this->lbounds[i] = bnds[i].l;
      this->dimension[i] = bnds[i].u - bnds[i].l + 1;
    }
    this->myData = data;
    this->refCount = nullptr;
  }
  template <class INT, typename std::enable_if< std::is_integral<INT>::value , int >::type = 0>
  YAKL_INLINE Array(char const * label, T * data, std::vector<INT> const &bnds) {
    #ifdef YAKL_DEBUG
      if ( bnds.size() < rank ) { yakl_throw("ERROR: Number of array bounds specified is < rank"); }
      if ( rank < 1 || rank > 8) { yakl_throw("ERROR: Invalid number of ranks. Must be between 1 and 8"); }
    #endif
    nullify();
    #ifdef YAKL_DEBUG
      this->myname = label;
    #endif
    for (int i=0; i < rank; i++) {
      this->lbounds[i] = 1;
      this->dimension[i] = bnds[i];
    }
    this->myData = data;
    this->refCount = nullptr;
  }


  /*
  COPY CONSTRUCTORS / FUNCTIONS
  This shares the pointers with another Array and increments the refCounter
  */
  YAKL_INLINE Array(Array<non_const_value_type,rank,myMem,styleFortran> const &rhs) {
    // This is a constructor, so no need to deallocate
    nullify();
    for (int i=0; i<rank; i++) {
      this->lbounds  [i] = rhs.lbounds  [i];
      this->dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      this->myname = rhs.myname;
    #endif
    this->myData   = rhs.myData;
    if (std::is_const<T>::value) {
      this->refCount = nullptr;
    } else {
      this->refCount = rhs.refCount;
      if (refCount != nullptr) {
        #if YAKL_CURRENTLY_ON_HOST()
          yakl_mtx_lock();
          (*this->refCount)++;
          yakl_mtx_unlock();
        #endif
      }
    }
  }
  YAKL_INLINE Array(Array<const_value_type,rank,myMem,styleFortran> const &rhs) {
    static_assert( std::is_const<T>::value , 
                   "ERROR: Cannot create non-const Array using const Array" );
    // This is a constructor, so no need to deallocate
    nullify();
    for (int i=0; i<rank; i++) {
      this->lbounds  [i] = rhs.lbounds  [i];
      this->dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      this->myname = rhs.myname;
    #endif
    this->myData   = rhs.myData;
    if (std::is_const<T>::value) {
      this->refCount = nullptr;
    } else {
      this->refCount = rhs.refCount;
      if (refCount != nullptr) {
        #if YAKL_CURRENTLY_ON_HOST()
          yakl_mtx_lock();
          (*this->refCount)++;
          yakl_mtx_unlock();
        #endif
      }
    }
  }


  YAKL_INLINE Array & operator=(Array<non_const_value_type,rank,myMem,styleFortran> const &rhs) {
    if (this == &rhs) { return *this; }
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
    for (int i=0; i<rank; i++) {
      this->lbounds  [i] = rhs.lbounds  [i];
      this->dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      this->myname = rhs.myname;
    #endif
    this->myData   = rhs.myData;
    if (std::is_const<T>::value) {
      this->refCount = nullptr;
    } else {
      this->refCount = rhs.refCount;
      if (refCount != nullptr) {
        #if YAKL_CURRENTLY_ON_HOST()
          yakl_mtx_lock();
          (*this->refCount)++;
          yakl_mtx_unlock();
        #endif
      }
    }
    return *this;
  }
  YAKL_INLINE Array & operator=(Array<const_value_type,rank,myMem,styleFortran> const &rhs) {
    static_assert( std::is_const<T>::value , 
                   "ERROR: Cannot create non-const Array using const Array" );
    if (this == &rhs) { return *this; }
    #if YAKL_CURRENTLY_ON_HOST()
      deallocate();
    #endif
    for (int i=0; i<rank; i++) {
      this->lbounds  [i] = rhs.lbounds  [i];
      this->dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      this->myname = rhs.myname;
    #endif
    this->myData   = rhs.myData;
    if (std::is_const<T>::value) {
      this->refCount = nullptr;
    } else {
    this->refCount = rhs.refCount;
      if (refCount != nullptr) {
        #if YAKL_CURRENTLY_ON_HOST()
          yakl_mtx_lock();
          (*this->refCount)++;
          yakl_mtx_unlock();
        #endif
      }
    }
    return *this;
  }


  /*
  MOVE CONSTRUCTORS
  This steals the pointers form the rhs instead of sharing and sets rhs pointers to nullptr.
  Therefore, no need to increment reference counter
  */
  YAKL_INLINE Array(Array &&rhs) {
    // This is a constructor, so no need to deallocate
    nullify();
    for (int i=0; i<rank; i++) {
      this->lbounds  [i] = rhs.lbounds  [i];
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
      this->lbounds  [i] = rhs.lbounds  [i];
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


  template <int N> inline Array<T,N,myMem,styleFortran> reshape(Bnds const &bnds) const {
    #ifdef YAKL_DEBUG
      if (bnds.size() != N) { yakl_throw("ERROR: new number of reshaped array dimensions does not match the templated rank"); }
      index_t totelems = 1;
      for (int i=0; i < N; i++) {
        totelems *= (bnds.u[i]-bnds.l[i]+1);
      }
      if (totelems != this->totElems()) { yakl_throw("ERROR: Total reshaped array elements is not consistent with this array"); }
    #endif
    Array<T,N,myMem,styleFortran> ret;
    for (int i=0; i < N; i++) {
      ret.dimension[i] = bnds.u[i] - bnds.l[i] + 1;
      ret.lbounds  [i] = bnds.l[i];
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


  inline Array<T,1,myMem,styleFortran> collapse(int lbnd=1) const {
    Array<T,1,myMem,styleFortran> ret;
    ret.dimension[0] = this->totElems();
    ret.lbounds  [0] = lbnd;
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
      this->myData = (T *) yaklAllocDevice( this->totElems()*sizeof(T) , label);
    } else {
      this->myData = new T[this->totElems()];
    }
  }


  template <class TLOC=T, typename std::enable_if< std::is_const<TLOC>::value , int >::type = 0>
  inline void deallocate() {
  }


  template <class TLOC=T, typename std::enable_if< ! std::is_const<TLOC>::value , int >::type = 0>
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





