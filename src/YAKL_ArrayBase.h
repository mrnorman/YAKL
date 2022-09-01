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
    index_t dimension[rank];  // Sizes of the 8 possible dimensions
    /** @private */
    int     * refCount;       // Pointer shared by multiple copies of this Array to keep track of allcation / free
    #ifdef YAKL_DEBUG
      /** @private */
      char const * myname;    // Label for debug printing. Only stored if debugging is turned on
    #endif
    #ifdef YAKL_ENABLE_STREAMS
      StreamList stream_dependencies;
    #else
      // This only exists to void extra data in the Array classes
      /** @private */
      struct StreamListDummy {
        static bool constexpr empty() { return true; }
        void push_back( Stream stream ) { }
        int  size() { return 0; }
        Stream operator[] (int i) { return Stream(); }
      };
      StreamListDummy stream_dependencies;
    #endif


    /** @brief Declare a dependency on the passed stream.
      * @details Upon deallocation, an event is placed in each stream this array depends on. The data pointer
      *          is not released from the pool until all dependent events complete. This avoids potential
      *          pointer aliasing of Arrays potentially being used simultaneous in different parallel streams.
      *          The pool allocator is non-blocking, so erroneous aliasing can occur if the user uses multiple
      *          streams, deallocates and allocates during runtime, and does not use this function. */
    void add_stream_dependency(Stream stream) {
      if constexpr (streams_enabled) {
        if (use_pool()) stream_dependencies.push_back(stream);
      }
    }


    /** @brief Declare a dependencies on the multiple streams at one time.
      * \copydetails add_stream_dependency */
    void add_stream_dependencies(std::vector<Stream> streams) {
      if constexpr (streams_enabled) {
        if (use_pool()) {
          for (int i=0; i < streams.size(); i++) { stream_dependencies.push_back(streams[i]); }
        }
      }
    }


    /** @private */
    void create_inform() {
      #ifdef YAKL_VERBOSE
        std::string msg = "Allocating ";
        if constexpr (myMem == memHost) {
          msg += std::string("host, ");
        } else {
          msg += std::string("device, ");
        }
        if constexpr (myStyle == styleC) {
          msg += std::string("C-style, ");
        } else {
          msg += std::string("Fortran-style, ");
        }
        msg += std::string("rank ") + std::to_string(rank) + std::string(" Array");
        msg += std::string(" of size ") + std::to_string(totElems()*sizeof(T)) + std::string(" bytes");
        verbose_inform(msg,this->label());
      #endif
    }


    /** @private */
    void destroy_inform() {
      #ifdef YAKL_VERBOSE
        std::string msg = "Deallocating ";
        if constexpr (myMem == memHost) {
          msg += std::string("host, ");
        } else {
          msg += std::string("device, ");
        }
        if constexpr (myStyle == styleC) {
          msg += std::string("C-style, ");
        } else {
          msg += std::string("Fortran-style, ");
        }
        msg += std::string("rank ") + std::to_string(rank) + std::string(" Array");
        verbose_inform(msg,this->label());
      #endif
    }


    /** @private */
    template <class ARR>
    void copy_inform(ARR const &dest) const {
      #ifdef YAKL_VERBOSE
        std::string msg = "Initiating ";
        if (myMem == memHost  ) msg += std::string("host to ");
        if (myMem == memDevice) msg += std::string("device to ");
        if (dest.get_memory_space() == memHost  ) msg += std::string("host memcpy of ");
        if (dest.get_memory_space() == memDevice) msg += std::string("device memcpy of ");
        msg += std::to_string(totElems()*sizeof(T)) + std::string(" bytes");
        if (label() != "") msg += " from Array labeled \"" + std::string(label()) + std::string("\"");
        if (dest.label() != "") msg += " to Array labeled \"" + std::string(dest.label()) + std::string("\"");
        verbose_inform(msg);
      #endif
    }


    // Deep copy this array's contents to another array that's on the host
    /** @brief [ASYNCHRONOUS] [DEEP_COPY] Copy this array's contents to a yakl::memHost array.
      * 
      * Arrays must have the same type and total number
      * of elements. No checking of rank, style, or dimensionality is performed. Both arrays must be allocated. 
      * `this` array may be in yakl::memHost or yakl::memDevice space. */
    template <int theirRank, int theirStyle>
    inline void deep_copy_to(Array<typename std::remove_cv<T>::type,theirRank,memHost,theirStyle> const &lhs , Stream stream = Stream()) const {
      #ifdef YAKL_VERBOSE
        copy_inform(lhs);
      #endif
      #ifdef YAKL_DEBUG
        if (this->totElems() != lhs.totElems()) { yakl_throw("ERROR: deep_copy_to with different number of elements"); }
        if (this->myData == nullptr || lhs.myData == nullptr) { yakl_throw("ERROR: deep_copy_to with nullptr"); }
      #endif
      if (myMem == memHost) { memcpy_host_to_host  ( lhs.myData , this->myData , this->totElems()          ); }
      else                  { memcpy_device_to_host( lhs.myData , this->myData , this->totElems() , stream ); }
      #ifdef YAKL_AUTO_FENCE
        fence();
      #endif
    }


    // Deep copy this array's contents to another array that's on the device
    /** @brief [ASYNCHRONOUS] [DEEP_COPY] Copy this array's contents to a yakl::memDevice array.
      * 
      * Arrays must have the same type and total number
      * of elements. No checking of rank, style, or dimensionality is performed. Both arrays must be allocated. 
      * `this` array may be in yakl::memHost or yakl::memDevice space. */
    template <int theirRank, int theirStyle>
    inline void deep_copy_to(Array<typename std::remove_cv<T>::type,theirRank,memDevice,theirStyle> const &lhs , Stream stream = Stream()) const {
      #ifdef YAKL_VERBOSE
        copy_inform(lhs);
      #endif
      #ifdef YAKL_DEBUG
        if (this->totElems() != lhs.totElems()) { yakl_throw("ERROR: deep_copy_to with different number of elements"); }
        if (this->myData == nullptr || lhs.myData == nullptr) { yakl_throw("ERROR: deep_copy_to with nullptr"); }
      #endif
      if (myMem == memHost) { memcpy_host_to_device  ( lhs.myData , this->myData , this->totElems() , stream ); }
      else                  { memcpy_device_to_device( lhs.myData , this->myData , this->totElems() , stream ); }
      #ifdef YAKL_AUTO_FENCE
        fence();
      #endif
    }


    /* ACCESSORS */
    /** @brief Returns the number of dimensions in this array object. */
    YAKL_INLINE int get_rank() const { return rank; }
    /** @brief Returns the total number of elements in this array object. */
    YAKL_INLINE index_t get_totElems() const {
      index_t tot = this->dimension[0];
      for (int i=1; i<rank; i++) { tot *= this->dimension[i]; }
      return tot;
    }
    /** @brief Returns the total number of elements in this array object. */
    YAKL_INLINE index_t get_elem_count() const { return get_totElems(); }
    /** @brief Returns the total number of elements in this array object. */
    YAKL_INLINE index_t totElems() const { return get_totElems(); }
    /** @brief Returns the total number of elements in this array object. */
    YAKL_INLINE index_t size() const { return get_totElems(); }
    /** @brief Returns the raw data pointer of this array object. */
    YAKL_INLINE T *data() const { return this->myData; }
    /** @brief Returns the raw data pointer of this array object. */
    YAKL_INLINE T *get_data() const { return this->myData; }
    /** @brief Always true. yakl::Array objects are always contiguous in memory with no padding. */
    YAKL_INLINE bool span_is_contiguous() const { return true; }
    /** @brief Returns whether this array object has is in an initialized / allocated state. */
    YAKL_INLINE bool initialized() const { return this->myData != nullptr; }
    /** @brief Returns this array object's string label if the `YAKL_DEBUG` CPP macro is defined. Otherwise, returns an empty string. */
    YAKL_INLINE int get_memory_space() const { return myMem == memHost ? memHost : memDevice; }
    const char* label() const {
      #ifdef YAKL_DEBUG
        return this->myname;
      #else
        return "\"Unlabeled: YAKL_DEBUG CPP macro not defined\"";
      #endif
    }
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
      #ifdef YAKL_VERBOSE
        this->create_inform();
      #endif
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
      yakl_mtx_lock();
      typedef typename std::remove_cv<T>::type T_non_const;
      T_non_const *data = const_cast<T_non_const *>(this->myData);
      if (this->refCount != nullptr) {
        (*(this->refCount))--;

        if (*this->refCount == 0) {
          #ifdef YAKL_VERBOSE
            destroy_inform();
          #endif
          delete this->refCount;
          this->refCount = nullptr;
          if (this->totElems() > 0) {
            if (myMem == memDevice) {
              if (streams_enabled && use_pool() && device_allocators_are_default && (! stream_dependencies.empty()) ) {
                std::vector<Event> event_dependencies;
                for (int i=0; i < stream_dependencies.size(); i++) {
                  event_dependencies.push_back( record_event(stream_dependencies[i]) );
                }
                pool.free_with_event_dependencies( data , event_dependencies , this->label() );
              } else {
                free_device(data,this->label());
              }
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
    /** @copydoc yakl::ArrayBase::deallocate() */
    template <class TLOC=T, typename std::enable_if< ! std::is_const<TLOC>::value , int >::type = 0>
    inline void deallocate() {
      yakl_mtx_lock();
      if (this->refCount != nullptr) {
        (*(this->refCount))--;

        if (*this->refCount == 0) {
          #ifdef YAKL_VERBOSE
            destroy_inform();
          #endif
          delete this->refCount;
          this->refCount = nullptr;
          if (this->totElems() > 0) {
            if (myMem == memDevice) {
              if (streams_enabled && use_pool() && device_allocators_are_default && (! stream_dependencies.empty()) ) {
                std::vector<Event> event_dependencies;
                for (int i=0; i < stream_dependencies.size(); i++) {
                  event_dependencies.push_back( record_event(stream_dependencies[i]) );
                }
                pool.free_with_event_dependencies( this->myData , event_dependencies , this->label() );
              } else {
                free_device(this->myData,this->label());
              }
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
        #ifdef YAKL_ENABLE_STREAMS
          fence();
        #endif
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

}


