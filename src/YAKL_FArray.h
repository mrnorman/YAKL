
#pragma once
// Included by YAKL_Array.h
// Inside the yakl namespace

template <class T, int rank, int myMem>
class Array<T,rank,myMem,styleFortran> : public FArrayBase<T,rank,myMem> {
public:

  typedef typename std::remove_cv<T>::type type;
  typedef          T value_type;
  typedef typename std::add_const<type>::type const_value_type;
  typedef typename std::remove_const<type>::type non_const_value_type;


  // Start off all constructors making sure the pointers are null
  YAKL_INLINE void nullify() {
    this->myData   = nullptr;
    this->refCount = nullptr;
    for (int i=0; i < rank; i++) { this->lbounds[i] = 1; this->dimension[i] = 0; }
    #ifdef YAKL_DEBUG
      this->myname="Uninitialized";
    #endif
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
  // Owned constructors
  YAKL_INLINE Array( char const* label , Bnd b1 ) : Array(label,Bnds(b1)) {}
  YAKL_INLINE Array( char const* label , Bnd b1 ,
                                         Bnd b2 ) : Array(label,Bnds(b1,b2)) {}
  YAKL_INLINE Array( char const* label , Bnd b1 ,
                                         Bnd b2 ,
                                         Bnd b3 ) : Array(label,Bnds(b1,b2,b3)) {}
  YAKL_INLINE Array( char const* label , Bnd b1 ,
                                         Bnd b2 ,
                                         Bnd b3 ,
                                         Bnd b4 ) : Array(label,Bnds(b1,b2,b3,b4)) {}
  YAKL_INLINE Array( char const* label , Bnd b1 ,
                                         Bnd b2 ,
                                         Bnd b3 ,
                                         Bnd b4 ,
                                         Bnd b5 ) : Array(label,Bnds(b1,b2,b3,b4,b5)) {}
  YAKL_INLINE Array( char const* label , Bnd b1 ,
                                         Bnd b2 ,
                                         Bnd b3 ,
                                         Bnd b4 ,
                                         Bnd b5 ,
                                         Bnd b6 ) : Array(label,Bnds(b1,b2,b3,b4,b5,b6)) {}
  YAKL_INLINE Array( char const* label , Bnd b1 ,
                                         Bnd b2 ,
                                         Bnd b3 ,
                                         Bnd b4 ,
                                         Bnd b5 ,
                                         Bnd b6 ,
                                         Bnd b7 ) : Array(label,Bnds(b1,b2,b3,b4,b5,b6,b7)) {}
  YAKL_INLINE Array( char const* label , Bnd b1 ,
                                         Bnd b2 ,
                                         Bnd b3 ,
                                         Bnd b4 ,
                                         Bnd b5 ,
                                         Bnd b6 ,
                                         Bnd b7 ,
                                         Bnd b8 ) : Array(label,Bnds(b1,b2,b3,b4,b5,b6,b7,b8)) {}
  YAKL_INLINE Array(char const * label, Bnds bnds) {
    nullify();
    #ifdef YAKL_DEBUG
      if ( bnds.size() < rank ) { yakl_throw("ERROR: Number of array bounds specified is < rank"); }
      if ( rank < 1 || rank > 8) { yakl_throw("ERROR: Invalid number of ranks. Must be between 1 and 8"); }
      this->myname = label;
    #endif
    #if YAKL_CURRENTLY_ON_HOST()
      this->deallocate();
    #endif
    for (int i=0; i < rank; i++) { this->lbounds[i] = bnds[i].l; this->dimension[i] = bnds[i].u - bnds[i].l + 1; }
    #if YAKL_CURRENTLY_ON_HOST()
      this->allocate(label);
    #endif
  }
  // Non-owned constructors
  YAKL_INLINE Array( char const *label , T *data, Bnd b1 ) : Array(label,data,Bnds(b1)) {}
  YAKL_INLINE Array( char const *label , T *data, Bnd b1 ,
                                                  Bnd b2 ) : Array(label,data,Bnds(b1,b2)) {}
  YAKL_INLINE Array( char const *label , T *data, Bnd b1 ,
                                                  Bnd b2 ,
                                                  Bnd b3 ) : Array(label,data,Bnds(b1,b2,b3)) {}
  YAKL_INLINE Array( char const *label , T *data, Bnd b1 ,
                                                  Bnd b2 ,
                                                  Bnd b3 ,
                                                  Bnd b4 ) : Array(label,data,Bnds(b1,b2,b3,b4)) {}
  YAKL_INLINE Array( char const *label , T *data, Bnd b1 ,
                                                  Bnd b2 ,
                                                  Bnd b3 ,
                                                  Bnd b4 ,
                                                  Bnd b5 ) : Array(label,data,Bnds(b1,b2,b3,b4,b5)) {}
  YAKL_INLINE Array( char const *label , T *data, Bnd b1 ,
                                                  Bnd b2 ,
                                                  Bnd b3 ,
                                                  Bnd b4 ,
                                                  Bnd b5 ,
                                                  Bnd b6 ) : Array(label,data,Bnds(b1,b2,b3,b4,b5,b6)) {}
  YAKL_INLINE Array( char const *label , T *data, Bnd b1 ,
                                                  Bnd b2 ,
                                                  Bnd b3 ,
                                                  Bnd b4 ,
                                                  Bnd b5 ,
                                                  Bnd b6 ,
                                                  Bnd b7 ) : Array(label,data,Bnds(b1,b2,b3,b4,b5,b6,b7)) {}
  YAKL_INLINE Array( char const *label , T *data, Bnd b1 ,
                                                  Bnd b2 ,
                                                  Bnd b3 ,
                                                  Bnd b4 ,
                                                  Bnd b5 ,
                                                  Bnd b6 ,
                                                  Bnd b7 ,
                                                  Bnd b8 ) : Array(label,data,Bnds(b1,b2,b3,b4,b5,b6,b7,b8)) {}
  YAKL_INLINE Array(char const *label, T *data, Bnds bnds) {
    nullify();
    #ifdef YAKL_DEBUG
      if ( bnds.size() < rank ) { yakl_throw("ERROR: Number of array bounds specified is < rank"); }
      if ( rank < 1 || rank > 8) { yakl_throw("ERROR: Invalid number of ranks. Must be between 1 and 8"); }
      if (data == nullptr) yakl_throw("ERROR: wrapping nullptr with a YAKL Array object");
      this->myname = label;
    #endif
    for (int i=0; i < rank; i++) { this->lbounds[i] = bnds[i].l; this->dimension[i] = bnds[i].u - bnds[i].l + 1; }
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
      this->lbounds  [i] = rhs.lbounds  [i]; this->dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      this->myname = rhs.myname;
    #endif
    this->myData   = rhs.myData;
    #if YAKL_CURRENTLY_ON_HOST()
      yakl_mtx_lock();
    #endif
    this->refCount = rhs.refCount;
    if (this->refCount != nullptr) {
      #if YAKL_CURRENTLY_ON_HOST()
        (*(this->refCount))++;
      #endif
    }
    #if YAKL_CURRENTLY_ON_HOST()
      yakl_mtx_unlock();
    #endif
  }
  YAKL_INLINE Array(Array<const_value_type,rank,myMem,styleFortran> const &rhs) {
    static_assert( std::is_const<T>::value , 
                   "ERROR: Cannot create non-const Array using const Array" );
    // This is a constructor, so no need to deallocate
    nullify();
    for (int i=0; i<rank; i++) {
      this->lbounds  [i] = rhs.lbounds  [i]; this->dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      this->myname = rhs.myname;
    #endif
    this->myData   = rhs.myData;
    #if YAKL_CURRENTLY_ON_HOST()
      yakl_mtx_lock();
    #endif
    this->refCount = rhs.refCount;
    if (this->refCount != nullptr) {
      #if YAKL_CURRENTLY_ON_HOST()
        (*(this->refCount))++;
      #endif
    }
    #if YAKL_CURRENTLY_ON_HOST()
      yakl_mtx_unlock();
    #endif
  }


  YAKL_INLINE Array & operator=(Array<non_const_value_type,rank,myMem,styleFortran> const &rhs) {
    if (this == &rhs) { return *this; }
    #if YAKL_CURRENTLY_ON_HOST()
      this->deallocate();
    #endif
    for (int i=0; i<rank; i++) {
      this->lbounds  [i] = rhs.lbounds  [i]; this->dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      this->myname = rhs.myname;
    #endif
    this->myData   = rhs.myData;
    #if YAKL_CURRENTLY_ON_HOST()
      yakl_mtx_lock();
    #endif
    this->refCount = rhs.refCount;
    if (this->refCount != nullptr) {
      #if YAKL_CURRENTLY_ON_HOST()
        (*(this->refCount))++;
      #endif
    }
    #if YAKL_CURRENTLY_ON_HOST()
      yakl_mtx_unlock();
    #endif
    return *this;
  }
  YAKL_INLINE Array & operator=(Array<const_value_type,rank,myMem,styleFortran> const &rhs) {
    static_assert( std::is_const<T>::value , 
                   "ERROR: Cannot create non-const Array using const Array" );
    if (this == &rhs) { return *this; }
    #if YAKL_CURRENTLY_ON_HOST()
      this->deallocate();
    #endif
    for (int i=0; i<rank; i++) {
      this->lbounds  [i] = rhs.lbounds  [i]; this->dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      this->myname = rhs.myname;
    #endif
    this->myData   = rhs.myData;
    #if YAKL_CURRENTLY_ON_HOST()
      yakl_mtx_lock();
    #endif
    this->refCount = rhs.refCount;
    if (this->refCount != nullptr) {
      #if YAKL_CURRENTLY_ON_HOST()
        (*(this->refCount))++;
      #endif
    }
    #if YAKL_CURRENTLY_ON_HOST()
      yakl_mtx_unlock();
    #endif
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
      this->lbounds  [i] = rhs.lbounds  [i]; this->dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      this->myname = rhs.myname;
    #endif
    this->myData   = rhs.myData;
    rhs.myData   = nullptr;

    this->refCount = rhs.refCount;
    rhs.refCount = nullptr;
  }


  YAKL_INLINE Array& operator=(Array &&rhs) {
    if (this == &rhs) { return *this; }
    #if YAKL_CURRENTLY_ON_HOST()
      this->deallocate();
    #endif
    for (int i=0; i<rank; i++) {
      this->lbounds  [i] = rhs.lbounds  [i]; this->dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      this->myname = rhs.myname;
    #endif
    this->myData   = rhs.myData;
    rhs.myData   = nullptr;

    this->refCount = rhs.refCount;
    rhs.refCount = nullptr;

    return *this;
  }


  /*
  DESTRUCTOR
  Decrement the refCounter, and if it's zero, deallocate and nullify.  
  */
  YAKL_INLINE ~Array() {
    #if YAKL_CURRENTLY_ON_HOST()
      this->deallocate();
    #endif
  }


  template <int N> inline Array<T,N,myMem,styleFortran> reshape(Bnds const &bnds) const {
    #ifdef YAKL_DEBUG
      if (! this->initialized()) { yakl_throw("ERROR: Trying to reshape an Array that hasn't been initialized"); }
      if (bnds.size() != N) { yakl_throw("ERROR: new number of reshaped array dimensions does not match the templated rank"); }
      index_t totelems = 1;
      for (int i=0; i < N; i++) {
        totelems *= (bnds.u[i]-bnds.l[i]+1);
      }
      if (totelems != this->totElems()) { yakl_throw("ERROR: Total reshaped array elements is not consistent with this array"); }
    #endif
    Array<T,N,myMem,styleFortran> ret;
    for (int i=0; i < N; i++) {
      ret.dimension[i] = bnds.u[i] - bnds.l[i] + 1;  ret.lbounds  [i] = bnds.l[i];
    }
    #ifdef YAKL_DEBUG
      ret.myname = this->myname;
    #endif
    ret.myData = this->myData;
    yakl_mtx_lock();
    ret.refCount = this->refCount;
    if (this->refCount != nullptr) {
      (*(this->refCount))++;
    }
    yakl_mtx_unlock();
    return ret;
  }


  inline Array<T,1,myMem,styleFortran> collapse(int lbnd=1) const {
    #ifdef YAKL_DEBUG
      if (! this->initialized()) { yakl_throw("ERROR: Trying to collapse an Array that hasn't been initialized"); }
    #endif
    Array<T,1,myMem,styleFortran> ret;
    ret.dimension[0] = this->totElems();  ret.lbounds  [0] = lbnd;
    #ifdef YAKL_DEBUG
      ret.myname = this->myname;
    #endif
    ret.myData = this->myData;
    yakl_mtx_lock();
    ret.refCount = this->refCount;
    if (this->refCount != nullptr) {
      (*(this->refCount))++;
    }
    yakl_mtx_unlock();
    return ret;
  }

};



