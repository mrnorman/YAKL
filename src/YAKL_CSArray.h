
#pragma once
// Included by YAKL_Array.h

namespace yakl {

  // This is a low-overhead class to represent a multi-dimensional C-style array with compile-time
  // known bounds placed on the stack of whatever context it is declared just like "int var[20];"
  // except with multiple dimensions, index checking, and printing

  /** @brief This creates a C-style "Stack Array" (CSArray) class, which is typedefined to yakl::SArray. It should
    *        be thought of as very similar to a C-style multi-dimensional array, `float arr[n1][n2][n3];`. That array
    *        as an `SArray` object would be created as `yakl::SArray<float,3,n1,n2,n3> arr;`, and it would be indexed
    *        as `arr(i1,i2,i3);`. For bounds checking, define the CPP macro `YAKL_DEBUG`. Dimensions sizes must be
    *        known at compile time, and data is placed on the stack of whatever context it is declared. When declared
    *        in a device `parallel_for` kernel, it is a thread-private array, meaning every thread has a separate copy
    *        of the array. 
    * @param T      Type of the yakl::SArray object
    * @param rank   Number of dimensions
    * @param D[0-3] Dimensions sizes. D1, D2, and D3 are optional template parameters
    * 
    * Creating these arrays is very cheap, but copying them does a deep copy every time and can be expensive.
    * yakl::SArray objects should be indexed with zero-based indices in row-major order (right-most index varies the fastest)
    */
  template <class T, int rank, unsigned D0, unsigned D1=1, unsigned D2=1, unsigned D3=1>
  class CSArray {
  public :

    /** @brief This is the type `T` without `const` and `volatile` modifiers */
    typedef typename std::remove_cv<T>::type       type;
    /** @brief This is the type `T` exactly as it was defined upon array object creation. */
    typedef          T                             value_type;
    /** @brief This is the type `T` with `const` added to it (if the original type has `volatile`, then so will this type. */
    typedef typename std::add_const<type>::type    const_value_type;
    /** @brief This is the type `T` with `const` removed from it (if the original type has `volatile`, then so will this type. */
    typedef typename std::remove_const<type>::type non_const_value_type;

    /** @private */
    static unsigned constexpr OFF0 = D3*D2*D1;
    /** @private */
    static unsigned constexpr OFF1 = D3*D2;
    /** @private */
    static unsigned constexpr OFF2 = D3;
    /** @private */
    static unsigned constexpr OFF3 = 1;

    /** @private */
    T mutable myData[D0*D1*D2*D3];

    // All copies are deep, so be wary of copies. Use references where possible
    /** @brief All copy and move constructors do a deep copy of all of the data, so they should be considered as possibly expensive */
    /// @{
    YAKL_INLINE CSArray() { }
    YAKL_INLINE CSArray           (CSArray      &&in) { for (uint i=0; i < totElems(); i++) { myData[i] = in.myData[i]; } }
    YAKL_INLINE CSArray           (CSArray const &in) { for (uint i=0; i < totElems(); i++) { myData[i] = in.myData[i]; } }
    YAKL_INLINE CSArray &operator=(CSArray      &&in) { for (uint i=0; i < totElems(); i++) { myData[i] = in.myData[i]; }; return *this; }
    YAKL_INLINE CSArray &operator=(CSArray const &in) { for (uint i=0; i < totElems(); i++) { myData[i] = in.myData[i]; }; return *this; }
    YAKL_INLINE ~CSArray() { }
    /// @}

    /** @brief Index the yakl::SArray object. Number of indices must match the rank of the array object. For bounds checking, 
      *        define the CPP macro `YAKL_DEBUG`. */
    /// @{
    YAKL_INLINE T &operator()(uint const i0) const {
      static_assert(rank==1,"ERROR: Improper number of dimensions specified in operator()");
      #ifdef YAKL_DEBUG
        #if YAKL_CURRENTLY_ON_HOST()
          if constexpr (rank >= 1) { if (i0>D0-1) { printf("CSArray i0 out of bounds (i0: %d; lb0: %d; ub0: %d)\n",i0,0,D0-1); yakl_throw(""); } }
        #else
          if constexpr (rank >= 1) { if (i0>D0-1) { yakl_throw("ERROR: CSArray index out of bounds"); } }
        #endif
      #endif
      return myData[i0];
    }
    YAKL_INLINE T &operator()(uint const i0, uint const i1) const {
      static_assert(rank==2,"ERROR: Improper number of dimensions specified in operator()");
      #ifdef YAKL_DEBUG
        #if YAKL_CURRENTLY_ON_HOST()
          if constexpr (rank >= 1) { if (i0>D0-1) { printf("CSArray i0 out of bounds (i0: %d; lb0: %d; ub0: %d)\n",i0,0,D0-1); yakl_throw(""); } }
          if constexpr (rank >= 2) { if (i1>D1-1) { printf("CSArray i1 out of bounds (i1: %d; lb1: %d; ub1: %d)\n",i1,0,D1-1); yakl_throw(""); } }
        #else
          if constexpr (rank >= 1) { if (i0>D0-1) { yakl_throw("ERROR: CSArray index out of bounds"); } }
          if constexpr (rank >= 2) { if (i1>D1-1) { yakl_throw("ERROR: CSArray index out of bounds"); } }
        #endif
      #endif
      return myData[i0*OFF0 + i1];
    }
    YAKL_INLINE T &operator()(uint const i0, uint const i1, uint const i2) const {
      static_assert(rank==3,"ERROR: Improper number of dimensions specified in operator()");
      #ifdef YAKL_DEBUG
        #if YAKL_CURRENTLY_ON_HOST()
          if constexpr (rank >= 1) { if (i0>D0-1) { printf("CSArray i0 out of bounds (i0: %d; lb0: %d; ub0: %d)\n",i0,0,D0-1); yakl_throw(""); } }
          if constexpr (rank >= 2) { if (i1>D1-1) { printf("CSArray i1 out of bounds (i1: %d; lb1: %d; ub1: %d)\n",i1,0,D1-1); yakl_throw(""); } }
          if constexpr (rank >= 3) { if (i2>D2-1) { printf("CSArray i2 out of bounds (i2: %d; lb2: %d; ub2: %d)\n",i2,0,D2-1); yakl_throw(""); } }
        #else
          if constexpr (rank >= 1) { if (i0>D0-1) { yakl_throw("ERROR: CSArray index out of bounds"); } }
          if constexpr (rank >= 2) { if (i1>D1-1) { yakl_throw("ERROR: CSArray index out of bounds"); } }
          if constexpr (rank >= 3) { if (i2>D2-1) { yakl_throw("ERROR: CSArray index out of bounds"); } }
        #endif
      #endif
      return myData[i0*OFF0 + i1*OFF1 + i2];
    }
    YAKL_INLINE T &operator()(uint const i0, uint const i1, uint const i2, uint const i3) const {
      static_assert(rank==4,"ERROR: Improper number of dimensions specified in operator()");
      #ifdef YAKL_DEBUG
        #if YAKL_CURRENTLY_ON_HOST()
          if constexpr (rank >= 1) { if (i0>D0-1) { printf("CSArray i0 out of bounds (i0: %d; lb0: %d; ub0: %d)\n",i0,0,D0-1); yakl_throw(""); } }
          if constexpr (rank >= 2) { if (i1>D1-1) { printf("CSArray i1 out of bounds (i1: %d; lb1: %d; ub1: %d)\n",i1,0,D1-1); yakl_throw(""); } }
          if constexpr (rank >= 3) { if (i2>D2-1) { printf("CSArray i2 out of bounds (i2: %d; lb2: %d; ub2: %d)\n",i2,0,D2-1); yakl_throw(""); } }
          if constexpr (rank >= 4) { if (i3>D3-1) { printf("CSArray i3 out of bounds (i3: %d; lb3: %d; ub3: %d)\n",i3,0,D3-1); yakl_throw(""); } }
        #else
          if constexpr (rank >= 1) { if (i0>D0-1) { yakl_throw("ERROR: CSArray index out of bounds"); } }
          if constexpr (rank >= 2) { if (i1>D1-1) { yakl_throw("ERROR: CSArray index out of bounds"); } }
          if constexpr (rank >= 3) { if (i2>D2-1) { yakl_throw("ERROR: CSArray index out of bounds"); } }
          if constexpr (rank >= 4) { if (i3>D3-1) { yakl_throw("ERROR: CSArray index out of bounds"); } }
        #endif
      #endif
      return myData[i0*OFF0 + i1*OFF1 + i2*OFF2 + i3];
    }
    /// @}


    /** @brief Assign a single arithmetic value to the entire array. */
    template <class TLOC , typename std::enable_if<std::is_arithmetic<TLOC>::value,int>::type = 0 >
    YAKL_INLINE void operator= (TLOC val) { for (int i=0 ; i < totElems() ; i++) { myData[i] = val; } }


    /** @brief Get the underlying raw data pointer */
    YAKL_INLINE T *data    () const { return myData; }
    /** @brief Get the underlying raw data pointer */
    YAKL_INLINE T *get_data() const { return myData; }
    /** @brief Get the total number of array elements */
    static unsigned constexpr totElems      () { return D3*D2*D1*D0; }
    /** @brief Get the total number of array elements */
    static unsigned constexpr get_totElems  () { return D3*D2*D1*D0; }
    /** @brief Get the total number of array elements */
    static unsigned constexpr get_elem_count() { return D3*D2*D1*D0; }
    /** @brief Get the number of dimensions */
    static unsigned constexpr get_rank      () { return rank; }
    /** @brief Always true. All YAKL arrays are contiguous with no padding. */
    static bool     constexpr span_is_contiguous() { return true; }
    /** @brief Always true. yakl::SArray objects are by default always initialized / allocated. */
    static bool     constexpr initialized() { return true; }


    /** @brief Print out the contents of this array. This should be called only from the host */
    inline friend std::ostream &operator<<(std::ostream& os, CSArray<T,rank,D0,D1,D2,D3> const &v) {
      for (uint i=0; i<totElems(); i++) { os << std::setw(12) << v.myData[i] << "\n"; }
      os << "\n";
      return os;
    }

    
    /** @brief Returns the dimensions of this array as a yakl::SArray object.
      *        You should use zero-based indexing on the returned SArray object. */
    YAKL_INLINE CSArray<uint,1,rank> get_dimensions() const {
      CSArray<uint,1,rank> ret;
      if constexpr (rank >= 1) ret(0) = D0;
      if constexpr (rank >= 2) ret(1) = D1;
      if constexpr (rank >= 3) ret(2) = D2;
      if constexpr (rank >= 4) ret(3) = D3;
      return ret;
    }
    /** @brief Returns the lower bound of each dimension of this array (which are always all zero) as a yakl::SArray object.
      *        You should use zero-based indexing on the returned yakl::SArray object. */
    YAKL_INLINE CSArray<uint,1,rank> get_lbounds() const {
      CSArray<uint,1,rank> ret;
      if constexpr (rank >= 1) ret(0) = 0;
      if constexpr (rank >= 2) ret(1) = 0;
      if constexpr (rank >= 3) ret(2) = 0;
      if constexpr (rank >= 4) ret(3) = 0;
      return ret;
    }
    /** @brief Returns the upper bound of each dimension of this array as a yakl::SArray object.
      *        You should use zero-based indexing on the returned yakl::SArray object. */
    YAKL_INLINE CSArray<uint,1,rank> get_ubounds() const {
      CSArray<uint,1,rank> ret;
      if constexpr (rank >= 1) ret(0) = D0-1;
      if constexpr (rank >= 2) ret(1) = D1-1;
      if constexpr (rank >= 3) ret(2) = D2-1;
      if constexpr (rank >= 4) ret(3) = D3-1;
      return ret;
    }

  };



  /** @brief Most often, codes use the type define yakl::SArray rather than yakl::CSArray */
  template <class T, int rank, unsigned D0, unsigned D1=1, unsigned D2=1, unsigned D3=1>
  using SArray = CSArray<T,rank,D0,D1,D2,D3>;

}


