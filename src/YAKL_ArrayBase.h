/**
 * @file
 *
 * Contains Array member data and functions common to both yakl::styleC and yakl::styleFortran array objects.
 */

#pragma once
// Included by YAKL_Array.h

namespace yakl {

  // This implements all functionality used by all dynamically allocated arrays
  /** @brief This class implements functionality common to both yakl::styleC and yakl::styleFortran `Array` objects.
    * 
    * @param T      Type of the array. For yakl::memHost array objects, this can generally be any type. For 
    *               yakl::memDevice array objects, this needs to be a type without a constructor, preferrably
    *               an arithmetic type.
    * @param rank   The number of dimensions for this array object.
    * @param myMem  The memory space for this array object: Either yakl::memHost or yakl::memDevice
    * @param myStyle The behavior of this array object: Either yakl::styleC or yakl::styleFortran
    */
  template <class T, int rank, int myMem, int myStyle>
  class ArrayBase {
  public:

    /** @brief This is the type `T` without `const` and `volatile` modifiers */
    typedef typename std::remove_cv<T>::type       type;
    /** @brief This is the type `T` exactly as it was defined upon array object creation. */
    typedef          T                             value_type;
    /** @brief This is the type `T` with `const` added to it (if the original type has `volatile`, then so will this type. */
    typedef typename std::add_const<type>::type    const_value_type;
    /** @brief This is the type `T` with `const` removed from it (if the original type has `volatile`, then so will this type. */
    typedef typename std::remove_const<type>::type non_const_value_type;

    /** @private */
    T       * myData;         // Pointer to the flattened internal data
    /** @private */
    size_t dimension[rank];  // Sizes of the 8 possible dimensions
    /** @private */
    int     * refCount;       // Pointer shared by multiple copies of this Array to keep track of allcation / free
    /** @private */
    char const * myname;    // Label for debug printing. Only stored if debugging is turned on


    // Deep copy this array's contents to another array that's on the host
    /** @brief [ASYNCHRONOUS] [DEEP_COPY] Copy this array's contents to a yakl::memHost array.
      * 
      * Arrays must have the same type and total number
      * of elements. No checking of rank, style, or dimensionality is performed. Both arrays must be allocated. 
      * `this` array may be in yakl::memHost or yakl::memDevice space. */
    template <int theirRank, int theirStyle>
    inline void deep_copy_to(Array<typename std::remove_cv<T>::type,theirRank,memHost,theirStyle> const &lhs ) const {
      #ifdef KOKKOS_DEBUG
        if (this->totElems() != lhs.totElems()) { Kokkos::abort("ERROR: deep_copy_to with different number of elements"); }
        if (this->myData == nullptr || lhs.myData == nullptr) { Kokkos::abort("ERROR: deep_copy_to with nullptr"); }
      #endif
      if (myMem == memHost) { memcpy_host_to_host  ( lhs.myData , this->myData , this->totElems() ); }
      else                  { memcpy_device_to_host( lhs.myData , this->myData , this->totElems() ); }
      #ifdef YAKL_AUTO_FENCE
        Kokkos::fence();
      #endif
    }


    // Deep copy this array's contents to another array that's on the device
    /** @brief [ASYNCHRONOUS] [DEEP_COPY] Copy this array's contents to a yakl::memDevice array.
      * 
      * Arrays must have the same type and total number
      * of elements. No checking of rank, style, or dimensionality is performed. Both arrays must be allocated. 
      * `this` array may be in yakl::memHost or yakl::memDevice space. */
    template <int theirRank, int theirStyle>
    inline void deep_copy_to(Array<typename std::remove_cv<T>::type,theirRank,memDevice,theirStyle> const &lhs ) const {
      #ifdef KOKKOS_DEBUG
        if (this->totElems() != lhs.totElems()) { Kokkos::abort("ERROR: deep_copy_to with different number of elements"); }
        if (this->myData == nullptr || lhs.myData == nullptr) { Kokkos::abort("ERROR: deep_copy_to with nullptr"); }
      #endif
      if (myMem == memHost) { memcpy_host_to_device  ( lhs.myData , this->myData , this->totElems() ); }
      else                  { memcpy_device_to_device( lhs.myData , this->myData , this->totElems() ); }
      #ifdef YAKL_AUTO_FENCE
        Kokkos::fence();
      #endif
    }


    /* ACCESSORS */
    /** @brief Returns the number of dimensions in this array object. */
    KOKKOS_INLINE_FUNCTION int get_rank() const { return rank; }
    /** @brief Returns the total number of elements in this array object. */
    KOKKOS_INLINE_FUNCTION size_t get_totElems() const {
      size_t tot = this->dimension[0];
      for (int i=1; i<rank; i++) { tot *= this->dimension[i]; }
      return tot;
    }
    /** @brief Returns the total number of elements in this array object. */
    KOKKOS_INLINE_FUNCTION size_t get_elem_count() const { return get_totElems(); }
    /** @brief Returns the total number of elements in this array object. */
    KOKKOS_INLINE_FUNCTION size_t totElems() const { return get_totElems(); }
    /** @brief Returns the total number of elements in this array object. */
    KOKKOS_INLINE_FUNCTION size_t size() const { return get_totElems(); }
    /** @brief Returns the raw data pointer of this array object. */
    KOKKOS_INLINE_FUNCTION T *data() const { return this->myData; }
    /** @brief Returns the raw data pointer of this array object. */
    KOKKOS_INLINE_FUNCTION T *get_data() const { return this->myData; }
    /** @brief Returns pointer to beginning of the data */
    KOKKOS_INLINE_FUNCTION T *begin() const { return this->myData; }
    /** @brief Returns pointer to end of the data */
    KOKKOS_INLINE_FUNCTION T *end() const { return begin() + size(); }
    /** @brief Always true. yakl::Array objects are always contiguous in memory with no padding. */
    KOKKOS_INLINE_FUNCTION bool span_is_contiguous() const { return true; }
    /** @brief Returns whether this array object has is in an initialized / allocated state. */
    KOKKOS_INLINE_FUNCTION bool initialized() const { return this->myData != nullptr; }
    /** @brief Returns this array object's string label if the `KOKKOS_DEBUG` CPP macro is defined. Otherwise, returns an empty string. */
    KOKKOS_INLINE_FUNCTION int get_memory_space() const { return myMem == memHost ? memHost : memDevice; }
    /** @brief Returns the object's string label as char const (Host only) */
    char const * label() const { return this->myname; }
    /** @brief Set the object's string label (Host only) */
    void set_label(std::string label) { this->myname = label.c_str(); }
    /** @brief Returns how many array objects share this pointer if owned; or `0` if unowned.
      * 
      * Returns the use count for this array object's data pointer. I.e., this is how many yakl::Array objects currently
      * share this data pointer. If this returns a value of `0`, that means that this array object is **not** being reference
      * counted, meaning it performed no allocation upon creation, will perform no deallocation upon destruction, and has
      * no control over whether the memory pointed to by the data pointer stays allocated or not. */
    inline int use_count() const {
      if (this->refCount != nullptr) { return *(this->refCount); }
      else                           { return 0;                 }
    }


    // Allocate the array and the reference counter (if owned)
    /** @private */
    template <class TLOC=T, typename std::enable_if< ! std::is_const<TLOC>::value , int >::type = 0>
    inline void allocate() {
      // static_assert( std::is_arithmetic<T>() || myMem == memHost , 
      //                "ERROR: You cannot use non-arithmetic types inside owned Arrays on the device" );
      yakl_mtx_lock();
      this->refCount = new int;
      (*(this->refCount)) = 1;
      if (myMem == memDevice) {
        this->myData = (T *) alloc_device( this->totElems()*sizeof(T) , this->label() );
      } else {
        this->myData = new T[this->totElems()];
      }
      yakl_mtx_unlock();
    }


    // Decrement the reference counter (if owned), and if it's zero after decrement, then deallocate the data
    // For const types, the pointer must be const casted to a non-const type before deallocation
    /** @brief If owned, decrement the reference counter; if ref counter reaches zero, deallocate memory; 
      *        If non-owned, does nothing.
      *
      *        If the reference counter reaches zero, meaning no other array objects
      *        are sharing this data pointer, the deallocate the data.
      *        This routine has the same effect as assigning this array object to
      *        an empty array object. This is safe to call even if this array object is not yet allocated. */
    template <class TLOC=T, typename std::enable_if< std::is_const<TLOC>::value , int >::type = 0>
    inline void deallocate() {
      typedef typename std::remove_cv<T>::type T_non_const;
      T_non_const *data = const_cast<T_non_const *>(this->myData);
      if (this->refCount != nullptr) {
        yakl_mtx_lock();
        (*(this->refCount))--;
        if (*this->refCount == 0) {
          delete this->refCount;
          this->refCount = nullptr;
          if (this->totElems() > 0) {
            if (myMem == memDevice) {
              free_device(data,this->label());
            } else {
              delete[] data;
            }
            this->myData = nullptr;
          }
        }
        yakl_mtx_unlock();
      }
    }


    // Decrement the reference counter (if owned), and if it's zero after decrement, then deallocate the data
    /** @copydoc yakl::ArrayBase::deallocate() */
    template <class TLOC=T, typename std::enable_if< ! std::is_const<TLOC>::value , int >::type = 0>
    inline void deallocate() {
      if (this->refCount != nullptr) {
        yakl_mtx_lock();
        (*(this->refCount))--;
        if (*this->refCount == 0) {
          delete this->refCount;
          this->refCount = nullptr;
          if (this->totElems() > 0) {
            if (myMem == memDevice) {
              free_device(this->myData,this->label());
            } else {
              delete[] this->myData;
            }
            this->myData = nullptr;
          }
        }
        yakl_mtx_unlock();
      }
    }


    // Print the array contents
    /** @brief Allows the user to `std::cout << this_array_object;`. This works even for yakl::memDevice array objects. */
    inline friend std::ostream &operator<<(std::ostream& os, Array<T,rank,myMem,myStyle> const &v) {
      os << "For Array labeled: " << v.label() << "\n";
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
        Kokkos::fence();
        local = from_dev;
      }
      for (size_t i=0; i<v.totElems(); i++) {
        os << local[i] << " ";
      }
      if (myMem == memDevice) {
        delete[] from_dev;
      }
      os << "\n";
      return os;
    }


  };

}


