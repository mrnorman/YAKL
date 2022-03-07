
#pragma once
// Included by YAKL_Array.h
// Inside the yakl namespace

template <class T, int rank, int myMem>
class FArrayBase : public ArrayBase<T,rank,myMem,styleFortran> {
public:

  typedef typename std::remove_cv<T>::type type;
  typedef          T value_type;
  typedef typename std::add_const<type>::type const_value_type;
  typedef typename std::remove_const<type>::type non_const_value_type;

  int lbounds[rank];  // Lower bounds

  /* ARRAY INDEXERS (FORTRAN index ordering)
  Return the element at the given index (either read-only or read-write)
  */
  YAKL_INLINE T &operator()(int i0) const {
    #ifdef YAKL_DEBUG
      check(1,i0);
    #endif
    index_t ind = (i0-this->lbounds[0]);
    return this->myData[ind];
  }
  YAKL_INLINE T &operator()(int i0, int i1) const {
    #ifdef YAKL_DEBUG
      check(2,i0,i1);
    #endif
    index_t ind =                       (i1-this->lbounds[1])  *
                   this->dimension[0] + (i0-this->lbounds[0]) ;
    return this->myData[ind];
  }
  YAKL_INLINE T &operator()(int i0, int i1, int i2) const {
    #ifdef YAKL_DEBUG
      check(3,i0,i1,i2);
    #endif
    index_t ind = (                      (i2-this->lbounds[2])  *
                    this->dimension[1] + (i1-this->lbounds[1]) )*
                    this->dimension[0] + (i0-this->lbounds[0]) ;
    return this->myData[ind];
  }
  YAKL_INLINE T &operator()(int i0, int i1, int i2, int i3) const {
    #ifdef YAKL_DEBUG
      check(4,i0,i1,i2,i3);
    #endif
    index_t ind = ((                      (i3-this->lbounds[3])  *
                     this->dimension[2] + (i2-this->lbounds[2]) )*
                     this->dimension[1] + (i1-this->lbounds[1]) )*
                     this->dimension[0] + (i0-this->lbounds[0]) ;
    return this->myData[ind];
  }
  YAKL_INLINE T &operator()(int i0, int i1, int i2, int i3, int i4) const {
    #ifdef YAKL_DEBUG
      check(5,i0,i1,i2,i3,i4);
    #endif
    index_t ind = (((                      (i4-this->lbounds[4])  *
                      this->dimension[3] + (i3-this->lbounds[3]) )*
                      this->dimension[2] + (i2-this->lbounds[2]) )*
                      this->dimension[1] + (i1-this->lbounds[1]) )*
                      this->dimension[0] + (i0-this->lbounds[0]) ;
    return this->myData[ind];
  }
  YAKL_INLINE T &operator()(int i0, int i1, int i2, int i3, int i4, int i5) const {
    #ifdef YAKL_DEBUG
      check(6,i0,i1,i2,i3,i4,i5);
    #endif
    index_t ind = ((((                      (i5-this->lbounds[5])  *
                       this->dimension[4] + (i4-this->lbounds[4]) )*
                       this->dimension[3] + (i3-this->lbounds[3]) )*
                       this->dimension[2] + (i2-this->lbounds[2]) )*
                       this->dimension[1] + (i1-this->lbounds[1]) )*
                       this->dimension[0] + (i0-this->lbounds[0]) ;
    return this->myData[ind];
  }
  YAKL_INLINE T &operator()(int i0, int i1, int i2, int i3, int i4, int i5, int i6) const {
    #ifdef YAKL_DEBUG
      check(7,i0,i1,i2,i3,i4,i5,i6);
    #endif
    index_t ind = (((((                      (i6-this->lbounds[6])  *
                        this->dimension[5] + (i5-this->lbounds[5]) )*
                        this->dimension[4] + (i4-this->lbounds[4]) )*
                        this->dimension[3] + (i3-this->lbounds[3]) )*
                        this->dimension[2] + (i2-this->lbounds[2]) )*
                        this->dimension[1] + (i1-this->lbounds[1]) )*
                        this->dimension[0] + (i0-this->lbounds[0]) ;
    return this->myData[ind];
  }
  YAKL_INLINE T &operator()(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7) const {
    #ifdef YAKL_DEBUG
      check(8,i0,i1,i2,i3,i4,i5,i6,i7);
    #endif
    index_t ind = ((((((                      (i7-this->lbounds[7])  *
                         this->dimension[6] + (i6-this->lbounds[6]) )*
                         this->dimension[5] + (i5-this->lbounds[5]) )*
                         this->dimension[4] + (i4-this->lbounds[4]) )*
                         this->dimension[3] + (i3-this->lbounds[3]) )*
                         this->dimension[2] + (i2-this->lbounds[2]) )*
                         this->dimension[1] + (i1-this->lbounds[1]) )*
                         this->dimension[0] + (i0-this->lbounds[0]) ;
    return this->myData[ind];
  }


  YAKL_INLINE void check(int rank_in, int i0, int i1=1, int i2=1, int i3=1, int i4=1, int i5=1, int i6=1,
                         int i7=1) const {
    if ( (!this->initialized()) ||
         (rank != rank_in) ||
         (rank >= 1 && (i0 < this->lbounds[0] || i0 >= this->lbounds[0]+this->dimension[0])) ||
         (rank >= 2 && (i1 < this->lbounds[1] || i1 >= this->lbounds[1]+this->dimension[1])) ||
         (rank >= 3 && (i2 < this->lbounds[2] || i2 >= this->lbounds[2]+this->dimension[2])) ||
         (rank >= 4 && (i3 < this->lbounds[3] || i3 >= this->lbounds[3]+this->dimension[3])) ||
         (rank >= 5 && (i4 < this->lbounds[4] || i4 >= this->lbounds[4]+this->dimension[4])) ||
         (rank >= 6 && (i5 < this->lbounds[5] || i5 >= this->lbounds[5]+this->dimension[5])) ||
         (rank >= 7 && (i6 < this->lbounds[6] || i6 >= this->lbounds[6]+this->dimension[6])) ||
         (rank >= 8 && (i7 < this->lbounds[7] || i7 >= this->lbounds[7]+this->dimension[7])) ) {
      print_error_message(rank,i0,i1,i2,i3,i4,i5,i6,i7);
    }
    #if defined(YAKL_SEPARATE_MEMORY_SPACE) && YAKL_CURRENTLY_ON_DEVICE()
      if (myMem == memHost) yakl_throw("ERROR: Accessing host memory on the device");
    #endif
    #if defined(YAKL_SEPARATE_MEMORY_SPACE) && YAKL_CURRENTLY_ON_HOST() && !defined(YAKL_MANAGED_MEMORY)
      if (myMem == memDevice) yakl_throw("ERROR: Accessing device memory on the host without managed memory");
    #endif

  }


  // if this function gets called, then there was definitely an error
  YAKL_INLINE void print_error_message(int rank_in, int i0, int i1, int i2, int i3, int i4, int i5, int i6,
                                       int i7) const {
    #ifdef YAKL_DEBUG
      #ifndef YAKL_SEPARATE_MEMORY_SPACE
        std::cerr << "For Array labeled: " << this->myname << ":" << std::endl;
        if (!this->initialized()) {
          yakl_throw("ERROR: operator() called on an Array that hasn't been allocated");
        }
        if (rank_in != rank) {
          std::cerr << "ERROR: operator() called with " << rank_in << " dimensions, but Array has "
                                                        << rank << " dimensions." << std::endl;
          yakl_throw("");
        }

        std::string msg1 = " is out of bounds. Value: ";
        std::string msg2 = "; Bounds: (";
        if (rank >= 1 && (i0 < this->lbounds[0] || i0 >= this->lbounds[0]+this->dimension[0])) {
          std::cerr << "Index 1 of " << rank << msg1 << i0 << msg2 << this->lbounds[0] << ","
                    << this->lbounds[0]+this->dimension[0]-1 << ")" << std::endl;
        }
        if (rank >= 2 && (i1 < this->lbounds[1] || i1 >= this->lbounds[1]+this->dimension[1])) {
          std::cerr << "Index 2 of " << rank << msg1 << i1 << msg2 << this->lbounds[1] << ","
                    << this->lbounds[1]+this->dimension[1]-1 << ")" << std::endl;
        }
        if (rank >= 3 && (i2 < this->lbounds[2] || i2 >= this->lbounds[2]+this->dimension[2])) {
          std::cerr << "Index 3 of " << rank << msg1 << i2 << msg2 << this->lbounds[2] << ","
                    << this->lbounds[2]+this->dimension[2]-1 << ")" << std::endl;
        }
        if (rank >= 4 && (i3 < this->lbounds[3] || i3 >= this->lbounds[3]+this->dimension[3])) {
          std::cerr << "Index 4 of " << rank << msg1 << i3 << msg2 << this->lbounds[3] << ","
                    << this->lbounds[3]+this->dimension[3]-1 << ")" << std::endl;
        }
        if (rank >= 5 && (i4 < this->lbounds[4] || i4 >= this->lbounds[4]+this->dimension[4])) {
          std::cerr << "Index 5 of " << rank << msg1 << i4 << msg2 << this->lbounds[4] << ","
                    << this->lbounds[4]+this->dimension[4]-1 << ")" << std::endl;
        }
        if (rank >= 6 && (i5 < this->lbounds[5] || i5 >= this->lbounds[5]+this->dimension[5])) {
          std::cerr << "Index 6 of " << rank << msg1 << i5 << msg2 << this->lbounds[5] << ","
                    << this->lbounds[5]+this->dimension[5]-1 << ")" << std::endl;
        }
        if (rank >= 7 && (i6 < this->lbounds[6] || i6 >= this->lbounds[6]+this->dimension[6])) {
          std::cerr << "Index 7 of " << rank << msg1 << i6 << msg2 << this->lbounds[6] << ","
                    << this->lbounds[6]+this->dimension[6]-1 << ")" << std::endl;
        }
        if (rank >= 8 && (i7 < this->lbounds[7] || i7 >= this->lbounds[7]+this->dimension[7])) {
          std::cerr << "Index 8 of " << rank << msg1 << i7 << msg2 << this->lbounds[7] << ","
                    << this->lbounds[7]+this->dimension[7]-1 << ")" << std::endl;
        }
        yakl_throw("");
      #endif
      yakl_throw("Error: one or more of the following has occurred: "
                 "(1) operator() called while array is not initialized; "
                 "(2) operator() called with the wrong number of dimensions; AND / OR "
                 "(3) operator() called with an index that is out of bounds.");
    #endif
  }


  // Array slicing
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( Dims const &dims ) const {
    #ifdef YAKL_DEBUG
      if (rank != dims.size()) {
        #ifndef YAKL_SEPARATE_MEMORY_SPACE
          std::cerr << "For Array named " << this->myname << ":  ";
        #endif
        yakl_throw("ERROR: rank must be equal to dims.size()");
      }
      for (int i=N; i<rank; i++) {
        if (dims.data[i] < this->lbounds[i] || dims.data[i] >= this->lbounds[i]+this->dimension[i] ) {
        #ifndef YAKL_SEPARATE_MEMORY_SPACE
            std::cerr << "For Array named " << this->myname << ":  ";
          #endif
          yakl_throw("ERROR: One of the slicing dimension dimensions is out of bounds");
        }
      }
      if (! this->initialized()) {
        #ifndef YAKL_SEPARATE_MEMORY_SPACE
          std::cerr << "For Array named " << this->myname << ":  ";
        #endif
        yakl_throw("ERROR: calling slice() on an Array that hasn't been allocated");
      }
    #endif
    Array<T,N,myMem,styleFortran> ret;
    index_t offset = 1;
    for (int i=0; i<N; i++) {
      ret.dimension[i] = this->dimension[i];
      ret.lbounds  [i] = this->lbounds  [i];
      offset *= this->dimension[i];
    }
    index_t retOff = 0;
    for (int i=N; i<rank; i++) {
      retOff += (dims.data[i]-this->lbounds[i])*offset;
      offset *= this->dimension[i];
    }
    ret.myData = &(this->myData[retOff]);
    return ret;
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0 ) const {
    return slice<N>( Dims(i0) );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0, int i1 ) const {
    return slice<N>( Dims(i0,i1) );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0, int i1, int i2 ) const {
    return slice<N>( Dims(i0,i1,i2) );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0, int i1, int i2, int i3 ) const {
    return slice<N>( Dims(i0,i1,i2,i3) );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0, int i1, int i2, int i3,
                                                                    int i4 ) const {
    return slice<N>( Dims(i0,i1,i2,i3,i4) );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0, int i1, int i2, int i3,
                                                                    int i4, int i5 ) const {
    return slice<N>( Dims(i0,i1,i2,i3,i4,i5) );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0, int i1, int i2, int i3,
                                                                    int i4, int i5, int i6 ) const {
    return slice<N>( Dims(i0,i1,i2,i3,i4,i5,i6) );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0, int i1, int i2, int i3,
                                                                    int i4, int i5, int i6, int i7 ) const {
    return slice<N>( Dims(i0,i1,i2,i3,i4,i5,i6,i7) );
  }


  // Create a host copy of this array. Even if the array exists on the host, a deep copy to a separate
  // object is still performed to avoid any potential bugs when the user expects this behavior
  inline Array<T,rank,memHost,styleFortran> createHostCopy() const {
    #ifdef YAKL_DEBUG
      if (! this->initialized()) {
        #ifndef YAKL_SEPARATE_MEMORY_SPACE
          std::cerr << "For Array named " << this->myname << ":  ";
        #endif
        yakl_throw("Error: createHostCopy() called on an Array that hasn't been allocated.");
      }
    #endif
    // If this Array is of const type, then we need to use non-const when allocating, then cast it to const aterward
    typedef typename std::remove_cv<T>::type T_NONCONST;
    Array<T_NONCONST,rank,memHost,styleFortran> ret;  // nullified + owned == true
    for (int i=0; i<rank; i++) {
      ret.lbounds  [i] = this->lbounds  [i];
      ret.dimension[i] = this->dimension[i];
    }
    #ifdef YAKL_DEBUG
      ret.myname = this->myname;
    #endif
    ret.allocate();
    if (myMem == memHost) {
      memcpy_host_to_host( ret.myData , this->myData , this->totElems() );
    } else {
      memcpy_device_to_host( ret.myData , this->myData , this->totElems() );
    }
    fence();
    return Array<T,rank,memHost,styleFortran>(ret);
  }


  // Create a device copy of this array. Even if the array exists on the host, a deep copy to a separate
  // object is still performed to avoid any potential bugs when the user expects this behavior
  inline Array<T,rank,memDevice,styleFortran> createDeviceCopy() const {
    #ifdef YAKL_DEBUG
      if (! this->initialized()) {
        #ifndef YAKL_SEPARATE_MEMORY_SPACE
          std::cerr << "For Array named " << this->myname << ":  ";
        #endif
        yakl_throw("Error: createHostCopy() called on an Array that hasn't been allocated.");
      }
    #endif
    // If this Array is of const type, then we need to use non-const when allocating, then cast it to const aterward
    typedef typename std::remove_cv<T>::type T_NONCONST;
    Array<T_NONCONST,rank,memDevice,styleFortran> ret;  // nullified + owned == true
    for (int i=0; i<rank; i++) {
      ret.lbounds  [i] = this->lbounds  [i];
      ret.dimension[i] = this->dimension[i];
    }
    #ifdef YAKL_DEBUG
      ret.myname = this->myname;
    #endif
    ret.allocate();
    if (myMem == memHost) {
      memcpy_host_to_device( ret.myData , this->myData , this->totElems() );
    } else {
      memcpy_device_to_device( ret.myData , this->myData , this->totElems() );
    }
    fence();
    return Array<T,rank,memDevice,styleFortran>(ret);
  }


  /* ACCESSORS */
  YAKL_INLINE FSArray<index_t,1,SB<rank>> get_dimensions() const {
    FSArray<index_t,1,SB<rank>> ret;
    for (int i=0; i<rank; i++) { ret(i+1) = this->dimension[i]; }
    return ret;
  }
  YAKL_INLINE FSArray<int,1,SB<rank>> get_lbounds() const {
    FSArray<int,1,SB<rank>> ret;
    for (int i=0; i<rank; i++) { ret(i+1) = this->lbounds[i]; }
    return ret;
  }
  YAKL_INLINE FSArray<int,1,SB<rank>> get_ubounds() const {
    FSArray<int,1,SB<rank>> ret;
    for (int i=0; i<rank; i++) { ret(i+1) = this->lbounds[i]+this->dimension[i]-1; }
    return ret;
  }
  YAKL_INLINE index_t extent( int dim ) const {
    return this->dimension[dim];
  }


};

