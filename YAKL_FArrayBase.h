
#pragma once

template <class T, int rank, int myMem, int myAttr>
class FArrayBase : public ArrayBase<T,rank,myMem,styleFortran,myAttr> {
public:

  int lbounds[rank];  // Lower bounds

  /* ARRAY INDEXERS (FORTRAN index ordering)
  Return the element at the given index (either read-only or read-write)
  */
  YAKL_INLINE T &operator()(int const i0) const {
    #ifdef YAKL_DEBUG
      if ( rank != 1 || i0 < this->lbounds[0] || i0 >= this->lbounds[0] + this->dimension[0] ) { indexing_check(1,i0); };
    #endif
    index_t ind = (i0-this->lbounds[0]);
    return this->myData[ind];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1) const {
    #ifdef YAKL_DEBUG
      if ( rank != 2 || i0 < this->lbounds[0] || i0 >= this->lbounds[0] + this->dimension[0] ||
                        i1 < this->lbounds[1] || i1 >= this->lbounds[1] + this->dimension[1] ) { indexing_check(2,i0,i1); };
    #endif
    index_t ind =                 (i1-this->lbounds[1])  *
                   this->dimension[0] + (i0-this->lbounds[0]) ;
    return this->myData[ind];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1, int const i2) const {
    #ifdef YAKL_DEBUG
      if ( rank != 3 || i0 < this->lbounds[0] || i0 >= this->lbounds[0] + this->dimension[0] ||
                        i1 < this->lbounds[1] || i1 >= this->lbounds[1] + this->dimension[1] ||
                        i2 < this->lbounds[2] || i2 >= this->lbounds[2] + this->dimension[2] ) { indexing_check(3,i0,i1,i2); };
    #endif
    index_t ind = (                (i2-this->lbounds[2])  *
                    this->dimension[1] + (i1-this->lbounds[1]) )*
                    this->dimension[0] + (i0-this->lbounds[0]) ;
    return this->myData[ind];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1, int const i2, int const i3) const {
    #ifdef YAKL_DEBUG
      if ( rank != 4 || i0 < this->lbounds[0] || i0 >= this->lbounds[0] + this->dimension[0] ||
                        i1 < this->lbounds[1] || i1 >= this->lbounds[1] + this->dimension[1] ||
                        i2 < this->lbounds[2] || i2 >= this->lbounds[2] + this->dimension[2] ||
                        i3 < this->lbounds[3] || i3 >= this->lbounds[3] + this->dimension[3] ) { indexing_check(4,i0,i1,i2,i3); };
    #endif
    index_t ind = ((                (i3-this->lbounds[3])  *
                     this->dimension[2] + (i2-this->lbounds[2]) )*
                     this->dimension[1] + (i1-this->lbounds[1]) )*
                     this->dimension[0] + (i0-this->lbounds[0]) ;
    return this->myData[ind];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1, int const i2, int const i3, int const i4) const {
    #ifdef YAKL_DEBUG
      if ( rank != 5 || i0 < this->lbounds[0] || i0 >= this->lbounds[0] + this->dimension[0] ||
                        i1 < this->lbounds[1] || i1 >= this->lbounds[1] + this->dimension[1] ||
                        i2 < this->lbounds[2] || i2 >= this->lbounds[2] + this->dimension[2] ||
                        i3 < this->lbounds[3] || i3 >= this->lbounds[3] + this->dimension[3] ||
                        i4 < this->lbounds[4] || i4 >= this->lbounds[4] + this->dimension[4] ) { indexing_check(5,i0,i1,i2,i3,i4); };
    #endif
    index_t ind = (((                (i4-this->lbounds[4])  *
                      this->dimension[3] + (i3-this->lbounds[3]) )*
                      this->dimension[2] + (i2-this->lbounds[2]) )*
                      this->dimension[1] + (i1-this->lbounds[1]) )*
                      this->dimension[0] + (i0-this->lbounds[0]) ;
    return this->myData[ind];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1, int const i2, int const i3, int const i4, int const i5) const {
    #ifdef YAKL_DEBUG
      if ( rank != 6 || i0 < this->lbounds[0] || i0 >= this->lbounds[0] + this->dimension[0] ||
                        i1 < this->lbounds[1] || i1 >= this->lbounds[1] + this->dimension[1] ||
                        i2 < this->lbounds[2] || i2 >= this->lbounds[2] + this->dimension[2] ||
                        i3 < this->lbounds[3] || i3 >= this->lbounds[3] + this->dimension[3] ||
                        i4 < this->lbounds[4] || i4 >= this->lbounds[4] + this->dimension[4] ||
                        i5 < this->lbounds[5] || i5 >= this->lbounds[5] + this->dimension[5] ) { indexing_check(6,i0,i1,i2,i3,i4,i5); };
    #endif
    index_t ind = ((((                (i5-this->lbounds[5])  *
                       this->dimension[4] + (i4-this->lbounds[4]) )*
                       this->dimension[3] + (i3-this->lbounds[3]) )*
                       this->dimension[2] + (i2-this->lbounds[2]) )*
                       this->dimension[1] + (i1-this->lbounds[1]) )*
                       this->dimension[0] + (i0-this->lbounds[0]) ;
    return this->myData[ind];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1, int const i2, int const i3, int const i4, int const i5, int const i6) const {
    #ifdef YAKL_DEBUG
      if ( rank != 7 || i0 < this->lbounds[0] || i0 >= this->lbounds[0] + this->dimension[0] ||
                        i1 < this->lbounds[1] || i1 >= this->lbounds[1] + this->dimension[1] ||
                        i2 < this->lbounds[2] || i2 >= this->lbounds[2] + this->dimension[2] ||
                        i3 < this->lbounds[3] || i3 >= this->lbounds[3] + this->dimension[3] ||
                        i4 < this->lbounds[4] || i4 >= this->lbounds[4] + this->dimension[4] ||
                        i5 < this->lbounds[5] || i5 >= this->lbounds[5] + this->dimension[5] ||
                        i6 < this->lbounds[6] || i6 >= this->lbounds[6] + this->dimension[6] ) { indexing_check(7,i0,i1,i2,i3,i4,i5,i6); };
    #endif
    index_t ind = (((((                (i6-this->lbounds[6])  *
                        this->dimension[5] + (i5-this->lbounds[5]) )*
                        this->dimension[4] + (i4-this->lbounds[4]) )*
                        this->dimension[3] + (i3-this->lbounds[3]) )*
                        this->dimension[2] + (i2-this->lbounds[2]) )*
                        this->dimension[1] + (i1-this->lbounds[1]) )*
                        this->dimension[0] + (i0-this->lbounds[0]) ;
    return this->myData[ind];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1, int const i2, int const i3, int const i4, int const i5, int const i6, int const i7) const {
    #ifdef YAKL_DEBUG
      if ( rank != 8 || i0 < this->lbounds[0] || i0 >= this->lbounds[0] + this->dimension[0] ||
                        i1 < this->lbounds[1] || i1 >= this->lbounds[1] + this->dimension[1] ||
                        i2 < this->lbounds[2] || i2 >= this->lbounds[2] + this->dimension[2] ||
                        i3 < this->lbounds[3] || i3 >= this->lbounds[3] + this->dimension[3] ||
                        i4 < this->lbounds[4] || i4 >= this->lbounds[4] + this->dimension[4] ||
                        i5 < this->lbounds[5] || i5 >= this->lbounds[5] + this->dimension[5] ||
                        i6 < this->lbounds[6] || i6 >= this->lbounds[6] + this->dimension[6] ||
                        i7 < this->lbounds[7] || i7 >= this->lbounds[7] + this->dimension[7] ) { indexing_check(8,i0,i1,i2,i3,i4,i5,i6,i7); };
    #endif
    index_t ind = ((((((                (i7-this->lbounds[7])  *
                         this->dimension[6] + (i6-this->lbounds[6]) )*
                         this->dimension[5] + (i5-this->lbounds[5]) )*
                         this->dimension[4] + (i4-this->lbounds[4]) )*
                         this->dimension[3] + (i3-this->lbounds[3]) )*
                         this->dimension[2] + (i2-this->lbounds[2]) )*
                         this->dimension[1] + (i1-this->lbounds[1]) )*
                         this->dimension[0] + (i0-this->lbounds[0]) ;
    return this->myData[ind];
  }


  // if this function gets called, then there was definitely an error
  YAKL_INLINE void indexing_check(int rank_in, index_t i0 ,
                                          index_t i1=INDEX_MAX ,
                                          index_t i2=INDEX_MAX ,
                                          index_t i3=INDEX_MAX ,
                                          index_t i4=INDEX_MAX ,
                                          index_t i5=INDEX_MAX ,
                                          index_t i6=INDEX_MAX ,
                                          index_t i7=INDEX_MAX ) const {
    #ifdef YAKL_DEBUG
      #ifndef YAKL_SEPARATE_MEMORY_SPACE
        std::cerr << "For Array labeled: " << this->myname << ":" << std::endl;
        std::string msg1 = " is out of bounds. Value: ";
        std::string msg2 = "; Bounds: (";
        if (rank_in != rank) { std::cerr << "Indexing with the incorrect number of dimensions. " << std::endl; }
        if (rank >= 1 && (i0 < this->lbounds[0] || i0 >= this->lbounds[0]+this->dimension[0])) {
          std::cerr << "Index 1 of " << rank << msg1 << i0 << msg2 << this->lbounds[0] << "," << this->lbounds[0]+this->dimension[0]-1 << ")" << std::endl;
        }
        if (rank >= 2 && (i1 < this->lbounds[1] || i1 >= this->lbounds[1]+this->dimension[1])) {
          std::cerr << "Index 2 of " << rank << msg1 << i1 << msg2 << this->lbounds[1] << "," << this->lbounds[1]+this->dimension[1]-1 << ")" << std::endl;
        }
        if (rank >= 3 && (i2 < this->lbounds[2] || i2 >= this->lbounds[2]+this->dimension[2])) {
          std::cerr << "Index 3 of " << rank << msg1 << i2 << msg2 << this->lbounds[2] << "," << this->lbounds[2]+this->dimension[2]-1 << ")" << std::endl;
        }
        if (rank >= 4 && (i3 < this->lbounds[3] || i3 >= this->lbounds[3]+this->dimension[3])) {
          std::cerr << "Index 4 of " << rank << msg1 << i3 << msg2 << this->lbounds[3] << "," << this->lbounds[3]+this->dimension[3]-1 << ")" << std::endl;
        }
        if (rank >= 5 && (i4 < this->lbounds[4] || i4 >= this->lbounds[4]+this->dimension[4])) {
          std::cerr << "Index 5 of " << rank << msg1 << i4 << msg2 << this->lbounds[4] << "," << this->lbounds[4]+this->dimension[4]-1 << ")" << std::endl;
        }
        if (rank >= 6 && (i5 < this->lbounds[5] || i5 >= this->lbounds[5]+this->dimension[5])) {
          std::cerr << "Index 6 of " << rank << msg1 << i5 << msg2 << this->lbounds[5] << "," << this->lbounds[5]+this->dimension[5]-1 << ")" << std::endl;
        }
        if (rank >= 7 && (i6 < this->lbounds[6] || i6 >= this->lbounds[6]+this->dimension[6])) {
          std::cerr << "Index 7 of " << rank << msg1 << i6 << msg2 << this->lbounds[6] << "," << this->lbounds[6]+this->dimension[6]-1 << ")" << std::endl;
        }
        if (rank >= 8 && (i7 < this->lbounds[7] || i7 >= this->lbounds[7]+this->dimension[7])) {
          std::cerr << "Index 8 of " << rank << msg1 << i7 << msg2 << this->lbounds[7] << "," << this->lbounds[7]+this->dimension[7]-1 << ")" << std::endl;
        }
      #endif
      yakl_throw("Invalid Array Index Encountered");
    #endif
  }


  template <int N> YAKL_INLINE void slice( Dims const &dims , Array<T,N,myMem,styleFortran,attrUnmanaged> &store ) const {
    #ifdef YAKL_DEBUG
      if (rank != dims.size()) {
        yakl_throw( "ERROR: rank must be equal to dims.size()" );
      }
      for (int i=N; i<rank; i++) {
        if (dims.data[i] < this->lbounds[i] || dims.data[i] >= this->lbounds[i]+this->dimension[i] ) {
          yakl_throw( "ERROR: One of the slicing dimension dimensions is out of bounds" );
        }
      }
    #endif
    index_t offset = 1;
    for (int i=0; i<N; i++) {
      store.dimension[i] = this->dimension[i];
      store.lbounds  [i] = this->lbounds  [i];
      offset *= this->dimension[i];
    }
    index_t retOff = 0;
    for (int i=N; i<rank; i++) {
      retOff += (dims.data[i]-this->lbounds[i])*offset;
      offset *= this->dimension[i];
    }
    store.myData = &(this->myData[retOff]);
  }
  template <int N> YAKL_INLINE void slice( int i0 , Array<T,N,myMem,styleFortran,attrUnmanaged> &store ) const {
    slice( {i0} , store );
  }
  template <int N> YAKL_INLINE void slice( int i0, int i1 , Array<T,N,myMem,styleFortran,attrUnmanaged> &store ) const {
    slice( {i0,i1} , store );
  }
  template <int N> YAKL_INLINE void slice( int i0, int i1, int i2, Array<T,N,myMem,styleFortran,attrUnmanaged> &store ) const {
    slice( {i0,i1,i2} , store );
  }
  template <int N> YAKL_INLINE void slice( int i0, int i1, int i2, int i3, Array<T,N,myMem,styleFortran,attrUnmanaged> &store ) const {
    slice( {i0,i1,i2,i3} , store );
  }
  template <int N> YAKL_INLINE void slice( int i0, int i1, int i2, int i3, int i4, Array<T,N,myMem,styleFortran,attrUnmanaged> &store ) const {
    slice( {i0,i1,i2,i3,i4} , store );
  }
  template <int N> YAKL_INLINE void slice( int i0, int i1, int i2, int i3, int i4, int i5, Array<T,N,myMem,styleFortran,attrUnmanaged> &store ) const {
    slice( {i0,i1,i2,i3,i4,i5} , store );
  }
  template <int N> YAKL_INLINE void slice( int i0, int i1, int i2, int i3, int i4, int i5, int i6, Array<T,N,myMem,styleFortran,attrUnmanaged> &store ) const {
    slice( {i0,i1,i2,i3,i4,i5,i6} , store );
  }
  template <int N> YAKL_INLINE void slice( int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, Array<T,N,myMem,styleFortran,attrUnmanaged> &store ) const {
    slice( {i0,i1,i2,i3,i4,i5,i6,i7} , store );
  }


  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran,attrUnmanaged> slice( Dims const &dims ) const {
    Array<T,N,myMem,styleFortran,attrUnmanaged> ret;
    slice( dims , ret );
    return ret;
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran,attrUnmanaged> slice( int i0 ) const {
    return slice<N>( {i0} );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran,attrUnmanaged> slice( int i0, int i1 ) const {
    return slice<N>( {i0,i1} );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran,attrUnmanaged> slice( int i0, int i1, int i2 ) const {
    return slice<N>( {i0,i1,i2} );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran,attrUnmanaged> slice( int i0, int i1, int i2, int i3 ) const {
    return slice<N>( {i0,i1,i2,i3} );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran,attrUnmanaged> slice( int i0, int i1, int i2, int i3, int i4 ) const {
    return slice<N>( {i0,i1,i2,i3,i4} );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran,attrUnmanaged> slice( int i0, int i1, int i2, int i3, int i4, int i5 ) const {
    return slice<N>( {i0,i1,i2,i3,i4,i5} );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran,attrUnmanaged> slice( int i0, int i1, int i2, int i3, int i4, int i5, int i6 ) const {
    return slice<N>( {i0,i1,i2,i3,i4,i5,i6} );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran,attrUnmanaged> slice( int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7 ) const {
    return slice<N>( {i0,i1,i2,i3,i4,i5,i6,i7} );
  }


  inline Array<T,rank,memHost,styleFortran> createHostCopy() const {
    Array<T,rank,memHost,styleFortran> ret;  // nullified + owned == true
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
    return ret;
  }

  inline Array<T,rank,memDevice,styleFortran> createDeviceCopy() const {
    Array<T,rank,memDevice,styleFortran> ret;  // nullified + owned == true
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
    return ret;
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
  YAKL_INLINE index_t extent( int const dim ) const {
    return this->dimension[dim];
  }


};

