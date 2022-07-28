
#pragma once
// Included by YAKL_Array.h

namespace yakl {

  template <class T, int rank, int myMem>
  class CArrayBase : public ArrayBase<T,rank,myMem,styleC> {
  public:

    typedef typename std::remove_cv<T>::type       type;
    typedef          T                             value_type;
    typedef typename std::add_const<type>::type    const_value_type;
    typedef typename std::remove_const<type>::type non_const_value_type;

    /* ARRAY INDEXERS (FORTRAN index ordering)
    Return the element at the given index (either read-only or read-write)
    */
    YAKL_INLINE T &operator()(index_t i0) const {
      static_assert( rank == 1 , "ERROR: Indexing non-rank-1 array with 1 index" );
      #ifdef YAKL_DEBUG
        check(i0);
      #endif
      index_t ind = i0;
      return this->myData[ind];
    }
    YAKL_INLINE T &operator()(index_t i0, index_t i1) const {
      static_assert( rank == 2 , "ERROR: Indexing non-rank-2 array with 2 indices" );
      #ifdef YAKL_DEBUG
        check(i0,i1);
      #endif
      index_t ind = i0*this->dimension[1] + i1;
      return this->myData[ind];
    }
    YAKL_INLINE T &operator()(index_t i0, index_t i1, index_t i2) const {
      static_assert( rank == 3 , "ERROR: Indexing non-rank-3 array with 3 indices" );
      #ifdef YAKL_DEBUG
        check(i0,i1,i2);
      #endif
      index_t ind = (i0*this->dimension[1] + i1)*
                        this->dimension[2] + i2;
      return this->myData[ind];
    }
    YAKL_INLINE T &operator()(index_t i0, index_t i1, index_t i2, index_t i3) const {
      static_assert( rank == 4 , "ERROR: Indexing non-rank-4 array with 4 indices" );
      #ifdef YAKL_DEBUG
        check(i0,i1,i2,i3);
      #endif
      index_t ind = ((i0*this->dimension[1] + i1)*
                         this->dimension[2] + i2)*
                         this->dimension[3] + i3;
      return this->myData[ind];
    }
    YAKL_INLINE T &operator()(index_t i0, index_t i1, index_t i2, index_t i3, index_t i4) const {
      static_assert( rank == 5 , "ERROR: Indexing non-rank-5 array with 5 indices" );
      #ifdef YAKL_DEBUG
        check(i0,i1,i2,i3,i4);
      #endif
      index_t ind = (((i0*this->dimension[1] + i1)*
                          this->dimension[2] + i2)*
                          this->dimension[3] + i3)*
                          this->dimension[4] + i4;
      return this->myData[ind];
    }
    YAKL_INLINE T &operator()(index_t i0, index_t i1, index_t i2, index_t i3, index_t i4, index_t i5) const {
      static_assert( rank == 6 , "ERROR: Indexing non-rank-6 array with 6 indices" );
      #ifdef YAKL_DEBUG
        check(i0,i1,i2,i3,i4,i5);
      #endif
      index_t ind = ((((i0*this->dimension[1] + i1)*
                           this->dimension[2] + i2)*
                           this->dimension[3] + i3)*
                           this->dimension[4] + i4)*
                           this->dimension[5] + i5;
      return this->myData[ind];
    }
    YAKL_INLINE T &operator()(index_t i0, index_t i1, index_t i2, index_t i3, index_t i4, index_t i5, index_t i6) const {
      static_assert( rank == 7 , "ERROR: Indexing non-rank-7 array with 7 indices" );
      #ifdef YAKL_DEBUG
        check(i0,i1,i2,i3,i4,i5,i6);
      #endif
      index_t ind = (((((i0*this->dimension[1] + i1)*
                            this->dimension[2] + i2)*
                            this->dimension[3] + i3)*
                            this->dimension[4] + i4)*
                            this->dimension[5] + i5)*
                            this->dimension[6] + i6;
      return this->myData[ind];
    }
    YAKL_INLINE T &operator()(index_t i0, index_t i1, index_t i2, index_t i3, index_t i4, index_t i5, index_t i6,
                              index_t i7) const {
      static_assert( rank == 8 , "ERROR: Indexing non-rank-8 array with 8 indices" );
      #ifdef YAKL_DEBUG
        check(i0,i1,i2,i3,i4,i5,i6,i7);
      #endif
      index_t ind = ((((((i0*this->dimension[1] + i1)*
                             this->dimension[2] + i2)*
                             this->dimension[3] + i3)*
                             this->dimension[4] + i4)*
                             this->dimension[5] + i5)*
                             this->dimension[6] + i6)*
                             this->dimension[7] + i7;
      return this->myData[ind];
    }


    YAKL_INLINE void check(index_t i0, index_t i1=0, index_t i2=0, index_t i3=0, index_t i4=0, index_t i5=0,
                           index_t i6=0, index_t i7=0) const {
      #ifdef YAKL_DEBUG
        if (! this->initialized()) { yakl_throw("Error: Using operator() on an Array that isn't allocated"); }
        if constexpr (rank >= 1) { if (i0 >= this->dimension[0]) ind_out_bounds<0>(i0); }
        if constexpr (rank >= 2) { if (i1 >= this->dimension[1]) ind_out_bounds<1>(i1); }
        if constexpr (rank >= 3) { if (i2 >= this->dimension[2]) ind_out_bounds<2>(i2); }
        if constexpr (rank >= 4) { if (i3 >= this->dimension[3]) ind_out_bounds<3>(i3); }
        if constexpr (rank >= 5) { if (i4 >= this->dimension[4]) ind_out_bounds<4>(i4); }
        if constexpr (rank >= 6) { if (i5 >= this->dimension[5]) ind_out_bounds<5>(i5); }
        if constexpr (rank >= 7) { if (i6 >= this->dimension[6]) ind_out_bounds<6>(i6); }
        if constexpr (rank >= 8) { if (i7 >= this->dimension[7]) ind_out_bounds<7>(i7); }
        #if defined(YAKL_SEPARATE_MEMORY_SPACE) && YAKL_CURRENTLY_ON_DEVICE()
          if constexpr (myMem == memHost) yakl_throw("ERROR: host array being accessed in a device kernel");
        #endif
        #if defined(YAKL_SEPARATE_MEMORY_SPACE) && YAKL_CURRENTLY_ON_HOST() && !defined(YAKL_MANAGED_MEMORY)
          if constexpr (myMem == memDevice) {
            std::cerr << "ERROR: For Array labeled: " << this->myname << ":" << std::endl;
            std::cerr << "Device array being accessed on the host without managed memory turned on";
            yakl_throw("");
          }
        #endif
      #endif
    }


    // if this function gets called, then there was definitely an error
    template <int I>
    YAKL_INLINE void ind_out_bounds(index_t ind) const {
      #ifdef YAKL_DEBUG
        #if YAKL_CURRENTLY_ON_HOST()
          std::cerr << "ERROR: For Array labeled: " << this->myname << ":" << std::endl;
          std::cerr << "Index " << I+1 << " of " << rank << " is out of bounds.  Provided index: " << ind
                    << ".  Upper Bound: " << this->dimension[I]-1 << std::endl;
          yakl_throw("");
        #else
          yakl_throw("ERROR: Index out of bounds.");
        #endif
      #endif
    }


    // Array slicing
    template <int N> YAKL_INLINE Array<T,N,myMem,styleC> slice( Dims const &dims ) const {
      #ifdef YAKL_DEBUG
        if (rank != dims.size()) {
          #if YAKL_CURRENTLY_ON_HOST()
            std::cerr << "For Array named " << this->myname << ":  ";
          #endif
          yakl_throw("ERROR: slice rank must be equal to dims.size()");
        }
        for (int i = rank-1-N; i >= 0; i--) {
          if (dims.data[i] >= this->dimension[i]) {
            #if YAKL_CURRENTLY_ON_HOST()
              std::cerr << "For Array named " << this->myname << ":  ";
            #endif
            yakl_throw("ERROR: One of the slicing dimension dimensions is out of bounds");
          }
        }
        if (! this->initialized()) {
          #if YAKL_CURRENTLY_ON_HOST()
            std::cerr << "For Array named " << this->myname << ":  ";
          #endif
          yakl_throw("ERROR: calling slice() on an Array that hasn't been allocated");
        }
      #endif
      Array<T,N,myMem,styleC> ret;
      index_t offset = 1;
      for (int i = rank-1; i > rank-1-N; i--) {
        ret.dimension[i-(rank-N)] = this->dimension[i];
        offset *= this->dimension[i];
      }
      index_t retOff = 0;
      for (int i = rank-1-N; i >= 0; i--) {
        retOff += dims.data[i]*offset;
        offset *= this->dimension[i];
      }
      ret.myData = &(this->myData[retOff]);
      #if YAKL_CURRENTLY_ON_HOST()
        yakl_mtx_lock();
        ret.refCount = this->refCount;
        if (this->refCount != nullptr) {
          (*(this->refCount))++;
        }
        yakl_mtx_unlock();
      #endif
      return ret;
    }
    template <int N> YAKL_INLINE Array<T,N,myMem,styleC> slice( int i0 ) const {
      static_assert( rank == 1 , "ERROR: Calling slice() with 1 index on a non-rank-1 array" );
      static_assert( N <= rank , "ERROR: Calling slice() with more dimenions than this array's rank" );
      return slice<N>( Dims(i0) );
    }
    template <int N> YAKL_INLINE Array<T,N,myMem,styleC> slice( int i0, int i1 ) const {
      static_assert( rank == 2 , "ERROR: Calling slice() with 2 index on a non-rank-2 array" );
      static_assert( N <= rank , "ERROR: Calling slice() with more dimenions than this array's rank" );
      return slice<N>( Dims(i0,i1) );
    }
    template <int N> YAKL_INLINE Array<T,N,myMem,styleC> slice( int i0, int i1, int i2 ) const {
      static_assert( rank == 3 , "ERROR: Calling slice() with 3 index on a non-rank-3 array" );
      static_assert( N <= rank , "ERROR: Calling slice() with more dimenions than this array's rank" );
      return slice<N>( Dims(i0,i1,i2) );
    }
    template <int N> YAKL_INLINE Array<T,N,myMem,styleC> slice( int i0, int i1, int i2, int i3 ) const {
      static_assert( rank == 4 , "ERROR: Calling slice() with 4 index on a non-rank-4 array" );
      static_assert( N <= rank , "ERROR: Calling slice() with more dimenions than this array's rank" );
      return slice<N>( Dims(i0,i1,i2,i3) );
    }
    template <int N> YAKL_INLINE Array<T,N,myMem,styleC> slice( int i0, int i1, int i2, int i3, int i4 ) const {
      static_assert( rank == 5 , "ERROR: Calling slice() with 5 index on a non-rank-5 array" );
      static_assert( N <= rank , "ERROR: Calling slice() with more dimenions than this array's rank" );
      return slice<N>( Dims(i0,i1,i2,i3,i4) );
    }
    template <int N> YAKL_INLINE Array<T,N,myMem,styleC> slice( int i0, int i1, int i2, int i3, int i4, int i5 ) const {
      static_assert( rank == 6 , "ERROR: Calling slice() with 6 index on a non-rank-6 array" );
      static_assert( N <= rank , "ERROR: Calling slice() with more dimenions than this array's rank" );
      return slice<N>( Dims(i0,i1,i2,i3,i4,i5) );
    }
    template <int N> YAKL_INLINE Array<T,N,myMem,styleC> slice( int i0, int i1, int i2, int i3, int i4, int i5,
                                                                int i6 ) const {
      static_assert( rank == 7 , "ERROR: Calling slice() with 7 index on a non-rank-7 array" );
      static_assert( N <= rank , "ERROR: Calling slice() with more dimenions than this array's rank" );
      return slice<N>( Dims(i0,i1,i2,i3,i4,i5,i6) );
    }
    template <int N> YAKL_INLINE Array<T,N,myMem,styleC> slice( int i0, int i1, int i2, int i3, int i4, int i5, int i6,
                                                                int i7 ) const {
      static_assert( rank == 8 , "ERROR: Calling slice() with 8 index on a non-rank-8 array" );
      static_assert( N <= rank , "ERROR: Calling slice() with more dimenions than this array's rank" );
      return slice<N>( Dims(i0,i1,i2,i3,i4,i5,i6,i7) );
    }


    // Create a host copy of this array. Even if the array exists on the host, a deep copy to a separate
    // object is still performed to avoid any potential bugs when the user expects this behavior
    template <class TLOC=T>
    inline Array<TLOC,rank,memHost,styleC> createHostCopy() const {
      auto ret = createHostObject();
      if (myMem == memHost) { memcpy_host_to_host  ( ret.myData , this->myData , this->totElems() ); }
      else                  { memcpy_device_to_host( ret.myData , this->myData , this->totElems() ); }
      fence();
      return Array<TLOC,rank,memHost,styleC>(ret);
    }


    // Create a separately allocate host object with the same rank, memory space, and style
    template <class TLOC=typename std::remove_cv<T>::type>
    inline Array<typename std::remove_cv<TLOC>::type,rank,memHost,styleC> createHostObject() const {
      #ifdef YAKL_DEBUG
        if (! this->initialized()) {
          #if YAKL_CURRENTLY_ON_HOST()
            std::cerr << "For Array named " << this->myname << ":  ";
          #endif
          yakl_throw("Error: createHostObject() called on an Array that hasn't been allocated");
        }
      #endif
      // If this Array is of const type, then we need to use non-const when allocating, then cast it to const aterward
      Array<typename std::remove_cv<TLOC>::type,rank,memHost,styleC> ret;
      for (int i=0; i<rank; i++) { ret.dimension[i] = this->dimension[i]; }
      #ifdef YAKL_DEBUG
        ret.myname = this->myname;
      #endif
      ret.allocate();
      return ret;
    }


    // Create a device copy of this array. Even if the array exists on the host, a deep copy to a separate
    // object is still performed to avoid any potential bugs when the user expects this behavior
    template <class TLOC=T>
    inline Array<TLOC,rank,memDevice,styleC> createDeviceCopy() const {
      auto ret = createDeviceObject();
      if (myMem == memHost) { memcpy_host_to_device  ( ret.myData , this->myData , this->totElems() ); }
      else                  { memcpy_device_to_device( ret.myData , this->myData , this->totElems() ); }
      fence();
      return Array<TLOC,rank,memDevice,styleC>(ret);
    }


    // Create a separately allocate device object with the same rank, memory space, and style
    template <class TLOC=typename std::remove_cv<T>::type>
    inline Array<typename std::remove_cv<TLOC>::type,rank,memDevice,styleC> createDeviceObject() const {
      #ifdef YAKL_DEBUG
        if (! this->initialized()) {
          #if YAKL_CURRENTLY_ON_HOST()
            std::cerr << "For Array named " << this->myname << ":  ";
          #endif
          yakl_throw("Error: createDeviceObject() called on an Array that hasn't been allocated.");
        }
      #endif
      // If this Array is of const type, then we need to use non-const when allocating, then cast it to const aterward
      Array<typename std::remove_cv<TLOC>::type,rank,memDevice,styleC> ret;
      for (int i=0; i<rank; i++) { ret.dimension[i] = this->dimension[i]; }
      #ifdef YAKL_DEBUG
        ret.myname = this->myname;
      #endif
      ret.allocate();
      return ret;
    }


    /* ACCESSORS */
    YAKL_INLINE SArray<index_t,1,rank> get_dimensions() const {
      SArray<index_t,1,rank> ret;
      for (int i=0; i<rank; i++) { ret(i) = this->dimension[i]; }
      return ret;
    }
    YAKL_INLINE SArray<index_t,1,rank> get_lbounds() const {
      SArray<index_t,1,rank> ret;
      for (int i=0; i<rank; i++) { ret(i) = 0; }
      return ret;
    }
    YAKL_INLINE SArray<index_t,1,rank> get_ubounds() const {
      SArray<index_t,1,rank> ret;
      for (int i=0; i<rank; i++) { ret(i) = this->dimension[i]-1; }
      return ret;
    }
    YAKL_INLINE index_t extent( int dim ) const {
      return this->dimension[dim];
    }


  };

}


