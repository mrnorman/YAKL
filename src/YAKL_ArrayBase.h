
#pragma once
// Included by YAKL_Array.h
// Inside the yakl namespace

// This implements all functionality used by all dynamically allocated arrays

template <class T, int rank, int myMem, int myStyle>
class ArrayBase {
public:

  typedef typename std::remove_cv<T>::type type;
  typedef          T value_type;
  typedef typename std::add_const<type>::type const_value_type;
  typedef typename std::remove_const<type>::type non_const_value_type;

  T       * myData;         // Pointer to the flattened internal data
  index_t dimension[rank];  // Sizes of the 8 possible dimensions
  int     * refCount;       // Pointer shared by multiple copies of this Array to keep track of allcation / free
  #ifdef YAKL_DEBUG
    char const * myname;    // Label for debug printing. Only stored if debugging is turned on
  #endif


  // Deep copy this array's contents to another array that's on the host
  template <int theirRank, int theirStyle>
  inline void deep_copy_to(Array<typename std::remove_cv<T>::type,theirRank,memHost,theirStyle> &lhs) const {
    #ifdef YAKL_DEBUG
      if (this->totElems() != lhs.totElems()) {
        yakl_throw("ERROR: deep_copy_to with different number of elements");
      }
      if (this->myData == nullptr || lhs.myData == nullptr) {
        yakl_throw("ERROR: deep_copy_to with nullptr");
      }
    #endif
    if (myMem == memHost) {
      memcpy_host_to_host( lhs.myData , this->myData , this->totElems() );
    } else {
      memcpy_device_to_host( lhs.myData , this->myData , this->totElems() );
    }
  }


  // Deep copy this array's contents to another array that's on the device
  template <int theirRank, int theirStyle>
  inline void deep_copy_to(Array<typename std::remove_cv<T>::type,theirRank,memDevice,theirStyle> &lhs) const {
    #ifdef YAKL_DEBUG
      if (this->totElems() != lhs.totElems()) {
        yakl_throw("ERROR: deep_copy_to with different number of elements");
      }
      if (this->myData == nullptr || lhs.myData == nullptr) {
        yakl_throw("ERROR: deep_copy_to with nullptr");
      }
    #endif
    if (myMem == memHost) {
      memcpy_host_to_device( lhs.myData , this->myData , this->totElems() );
    } else {
      memcpy_device_to_device( lhs.myData , this->myData , this->totElems() );
    }
  }


  /* ACCESSORS */
  YAKL_INLINE int get_rank() const { return rank; }
  YAKL_INLINE index_t get_totElems() const {
    index_t tot = this->dimension[0];
    for (int i=1; i<rank; i++) { tot *= this->dimension[i]; }
    return tot;
  }
  YAKL_INLINE index_t get_elem_count() const { return get_totElems(); }
  YAKL_INLINE index_t totElems() const { return get_totElems(); }
  YAKL_INLINE T *data() const { return this->myData; }
  YAKL_INLINE T *get_data() const { return this->myData; }
  YAKL_INLINE bool span_is_contiguous() const { return true; }
  YAKL_INLINE bool initialized() const { return this->myData != nullptr; }
  const char* label() const {
    #ifdef YAKL_DEBUG
      return this->myname;
    #else
      return "";
    #endif
  }
  inline int use_count() const {
    if (this->refCount != nullptr) {
      return *(this->refCount);
    } else {
      return 0;
    }
  }


  // Allocate the array and the reference counter (if owned)
  template <class TLOC=T, typename std::enable_if< ! std::is_const<TLOC>::value , int >::type = 0>
  inline void allocate(char const * label = "") {
    // static_assert( std::is_arithmetic<T>() || myMem == memHost , 
    //                "ERROR: You cannot use non-arithmetic types inside owned Arrays on the device" );
    yakl_mtx_lock();
    this->refCount = new int;
    (*(this->refCount)) = 1;
    if (myMem == memDevice) {
      this->myData = (T *) yaklAllocDevice( this->totElems()*sizeof(T) , label );
    } else {
      this->myData = new T[this->totElems()];
    }
    yakl_mtx_unlock();
  }


  // Decrement the reference counter (if owned), and if it's zero after decrement, then deallocate the data
  // For const types, the pointer must be const casted to a non-const type before deallocation
  template <class TLOC=T, typename std::enable_if< std::is_const<TLOC>::value , int >::type = 0>
  inline void deallocate() {
    yakl_mtx_lock();
    typedef typename std::remove_cv<T>::type T_non_const;
    T_non_const *data = const_cast<T_non_const *>(this->myData);
    if (this->refCount != nullptr) {
      (*(this->refCount))--;

      if (*this->refCount == 0) {
        delete this->refCount;
        this->refCount = nullptr;
        if (this->totElems() > 0) {
          if (myMem == memDevice) {
            #ifdef YAKL_DEBUG
              yaklFreeDevice(data,this->myname);
            #else
              yaklFreeDevice(data,"");
            #endif
          } else {
            delete[] data;
          }
          this->myData = nullptr;
        }
      }

    }
    yakl_mtx_unlock();
  }


  // Decrement the reference counter (if owned), and if it's zero after decrement, then deallocate the data
  template <class TLOC=T, typename std::enable_if< ! std::is_const<TLOC>::value , int >::type = 0>
  inline void deallocate() {
    yakl_mtx_lock();
    if (this->refCount != nullptr) {
      (*(this->refCount))--;

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
    yakl_mtx_unlock();
  }


  // Print the array contents
  inline friend std::ostream &operator<<(std::ostream& os, Array<T,rank,myMem,myStyle> const &v) {
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
    const_value_type     *local = v.myData;
    non_const_value_type *from_dev;
    if (myMem == memDevice) {
      from_dev = new non_const_value_type[v.totElems()];
      memcpy_device_to_host( from_dev , v.myData , v.totElems() );
      fence();
      local = from_dev;
    }
    for (index_t i=0; i<v.totElems(); i++) {
      os << local[i] << " ";
    }
    if (myMem == memDevice) {
      delete[] from_dev;
    }
    os << "\n";
    return os;
  }


};

