
#pragma once
// Included by YAKL_Array.h

namespace yakl {

  // This is a low-overhead class to represent a multi-dimensional C-style array with compile-time
  // known bounds placed on the stack of whatever context it is declared just like "int var[20];"
  // except with multiple dimensions, index checking, and printing

  /** @brief C-style array on the stack similar in nature to, e.g., `float arr[ny][nx];`
    *
    * This creates a C-style "Stack Array" (CSArray) class, which is typedefined to yakl::SArray. It should
    * be thought of as very similar to a C-style multi-dimensional array, `float arr[n1][n2][n3];`. That array
    * as an `SArray` object would be created as `yakl::SArray<float,3,n1,n2,n3> arr;`, and it would be indexed
    * as `arr(i1,i2,i3);`. For bounds checking, define the CPP macro `KOKKOS_ENABLE_DEBUG`. Dimensions sizes must be
    * known at compile time, and data is placed on the stack of whatever context it is declared. When declared
    * in a device `parallel_for` kernel, it is a thread-private array, meaning every thread has a separate copy
    * of the array. 
    * 
    * @param T      Type of the yakl::SArray object
    * @param rank   Number of dimensions
    * @param D[0-3] Dimensions sizes. D1, D2, and D3 are optional template parameters
    * 
    * Creating these arrays is very cheap, but copying them does a deep copy every time and can be expensive.
    * yakl::SArray objects should be indexed with zero-based indices in row-major order (right-most index varies the fastest)
    */
  template <class T, int rank, size_t D0, size_t D1=1, size_t D2=1, size_t D3=1>
  class CSArray {
  protected:
    /** @private */
    static size_t constexpr OFF0 = D3*D2*D1;
    /** @private */
    static size_t constexpr OFF1 = D3*D2;
    /** @private */
    static size_t constexpr OFF2 = D3;
    /** @private */
    static size_t constexpr OFF3 = 1;
    /** @private */
    T mutable myData[D0*D1*D2*D3];

  public :

    /** @brief This is the type `T` without `const` and `volatile` modifiers */
    typedef typename std::remove_cv<T>::type       type;
    /** @brief This is the type `T` exactly as it was defined upon array object creation. */
    typedef          T                             value_type;
    /** @brief This is the type `T` with `const` added to it (if the original type has `volatile`, then so will this type). */
    typedef typename std::add_const<type>::type    const_value_type;
    /** @brief This is the type `T` with `const` removed from it (if the original type has `volatile`, then so will this type). */
    typedef typename std::remove_const<type>::type non_const_value_type;

    /** @brief No constructor arguments allowed */
    KOKKOS_INLINE_FUNCTION CSArray(T init_fill) { for (int i=0; i < size(); i++) { myData[i] = init_fill; } }
    CSArray()  = default;
    ~CSArray() = default;

    /** @brief Returns a reference to the indexed element (1-D).
      * @details Number of indices must match the rank of the array object. For bounds checking, define the CPP macro `KOKKOS_ENABLE_DEBUG`.
      * Always use zero-based indexing with row-major ordering (right-most index varying the fastest). */
    KOKKOS_INLINE_FUNCTION T &operator()(size_t const i0) const {
      static_assert(rank==1,"ERROR: Improper number of dimensions specified in operator()");
      #ifdef KOKKOS_ENABLE_DEBUG
          if constexpr (rank >= 1) { if (i0>D0-1) { KOKKOS_IF_ON_HOST( printf("CSArray i0 out of bounds (i0: %zu; lb0: %d; ub0: %zu)\n",i0,0,D0-1); ) } }
          if constexpr (rank >= 1) { if (i0>D0-1) { Kokkos::abort("ERROR: CSArray index out of bounds"); } }
      #endif
      return myData[i0];
    }
    /** @brief Returns a reference to the indexed element (2-D).
      * @details Number of indices must match the rank of the array object. For bounds checking, define the CPP macro `KOKKOS_ENABLE_DEBUG`.
      * Always use zero-based indexing with row-major ordering (right-most index varying the fastest). */
    KOKKOS_INLINE_FUNCTION T &operator()(size_t const i0, size_t const i1) const {
      static_assert(rank==2,"ERROR: Improper number of dimensions specified in operator()");
      #ifdef KOKKOS_ENABLE_DEBUG
        if constexpr (rank >= 1) { if (i0>D0-1) { KOKKOS_IF_ON_HOST( printf("CSArray i0 out of bounds (i0: %zu; lb0: %d; ub0: %zu)\n",i0,0,D0-1); ) } }
        if constexpr (rank >= 2) { if (i1>D1-1) { KOKKOS_IF_ON_HOST( printf("CSArray i1 out of bounds (i1: %zu; lb1: %d; ub1: %zu)\n",i1,0,D1-1); ) } }
        if constexpr (rank >= 1) { if (i0>D0-1) { Kokkos::abort("ERROR: CSArray index out of bounds"); } }
        if constexpr (rank >= 2) { if (i1>D1-1) { Kokkos::abort("ERROR: CSArray index out of bounds"); } }
      #endif
      return myData[i0*OFF0 + i1];
    }
    /** @brief Returns a reference to the indexed element (3-D).
      * @details Number of indices must match the rank of the array object. For bounds checking, define the CPP macro `KOKKOS_ENABLE_DEBUG`.
      * Always use zero-based indexing with row-major ordering (right-most index varying the fastest). */
    KOKKOS_INLINE_FUNCTION T &operator()(size_t const i0, size_t const i1, size_t const i2) const {
      static_assert(rank==3,"ERROR: Improper number of dimensions specified in operator()");
      #ifdef KOKKOS_ENABLE_DEBUG
        if constexpr (rank >= 1) { if (i0>D0-1) { KOKKOS_IF_ON_HOST( printf("CSArray i0 out of bounds (i0: %zu; lb0: %d; ub0: %zu)\n",i0,0,D0-1); ) } }
        if constexpr (rank >= 2) { if (i1>D1-1) { KOKKOS_IF_ON_HOST( printf("CSArray i1 out of bounds (i1: %zu; lb1: %d; ub1: %zu)\n",i1,0,D1-1); ) } }
        if constexpr (rank >= 3) { if (i2>D2-1) { KOKKOS_IF_ON_HOST( printf("CSArray i2 out of bounds (i2: %zu; lb2: %d; ub2: %zu)\n",i2,0,D2-1); ) } }
        if constexpr (rank >= 1) { if (i0>D0-1) { Kokkos::abort("ERROR: CSArray index out of bounds"); } }
        if constexpr (rank >= 2) { if (i1>D1-1) { Kokkos::abort("ERROR: CSArray index out of bounds"); } }
        if constexpr (rank >= 3) { if (i2>D2-1) { Kokkos::abort("ERROR: CSArray index out of bounds"); } }
      #endif
      return myData[i0*OFF0 + i1*OFF1 + i2];
    }
    /** @brief Returns a reference to the indexed element (4-D).
      * @details Number of indices must match the rank of the array object. For bounds checking, define the CPP macro `KOKKOS_ENABLE_DEBUG`.
      * Always use zero-based indexing with row-major ordering (right-most index varying the fastest). */
    KOKKOS_INLINE_FUNCTION T &operator()(size_t const i0, size_t const i1, size_t const i2, size_t const i3) const {
      static_assert(rank==4,"ERROR: Improper number of dimensions specified in operator()");
      #ifdef KOKKOS_ENABLE_DEBUG
        if constexpr (rank >= 1) { if (i0>D0-1) { KOKKOS_IF_ON_HOST( printf("CSArray i0 out of bounds (i0: %zu; lb0: %d; ub0: %zu)\n",i0,0,D0-1); ) } }
        if constexpr (rank >= 2) { if (i1>D1-1) { KOKKOS_IF_ON_HOST( printf("CSArray i1 out of bounds (i1: %zu; lb1: %d; ub1: %zu)\n",i1,0,D1-1); ) } }
        if constexpr (rank >= 3) { if (i2>D2-1) { KOKKOS_IF_ON_HOST( printf("CSArray i2 out of bounds (i2: %zu; lb2: %d; ub2: %zu)\n",i2,0,D2-1); ) } }
        if constexpr (rank >= 4) { if (i3>D3-1) { KOKKOS_IF_ON_HOST( printf("CSArray i3 out of bounds (i3: %zu; lb3: %d; ub3: %zu)\n",i3,0,D3-1); ) } }
        if constexpr (rank >= 1) { if (i0>D0-1) { Kokkos::abort("ERROR: CSArray index out of bounds"); } }
        if constexpr (rank >= 2) { if (i1>D1-1) { Kokkos::abort("ERROR: CSArray index out of bounds"); } }
        if constexpr (rank >= 3) { if (i2>D2-1) { Kokkos::abort("ERROR: CSArray index out of bounds"); } }
        if constexpr (rank >= 4) { if (i3>D3-1) { Kokkos::abort("ERROR: CSArray index out of bounds"); } }
      #endif
      return myData[i0*OFF0 + i1*OFF1 + i2*OFF2 + i3];
    }


    /** @brief Assign a single arithmetic value to the entire array. */
    template <class TLOC , typename std::enable_if<std::is_arithmetic<TLOC>::value,int>::type = 0 >
    KOKKOS_INLINE_FUNCTION void operator= (TLOC val) { for (int i=0 ; i < totElems() ; i++) { myData[i] = val; } }


    /** @brief Get the underlying raw data pointer */
    KOKKOS_INLINE_FUNCTION T *data    () const { return myData; }
    /** @brief Get the underlying raw data pointer */
    KOKKOS_INLINE_FUNCTION T *get_data() const { return myData; }
    /** @brief Returns pointer to beginning of the data */
    KOKKOS_INLINE_FUNCTION T *begin() const { return myData; }
    /** @brief Returns pointer to end of the data */
    KOKKOS_INLINE_FUNCTION T *end() const { return begin() + size(); }
    /** @brief Get the total number of array elements */
    static size_t constexpr totElems      () { return D3*D2*D1*D0; }
    /** @brief Get the total number of array elements */
    static size_t constexpr get_totElems  () { return D3*D2*D1*D0; }
    /** @brief Get the total number of array elements */
    static size_t constexpr size          () { return D3*D2*D1*D0; }
    /** @brief Get the total number of array elements */
    static size_t constexpr get_elem_count() { return D3*D2*D1*D0; }
    /** @brief Get the number of dimensions */
    static size_t constexpr get_rank      () { return rank; }
    /** @brief Always true. All YAKL arrays are contiguous with no padding. */
    static bool     constexpr span_is_contiguous() { return true; }
    /** @brief Always true. yakl::SArray objects are by default always initialized / allocated. */
    static bool     constexpr initialized() { return true; }


    /** @brief Print out the contents of this array. This should be called only from the host */
    inline friend std::ostream &operator<<(std::ostream& os, CSArray<T,rank,D0,D1,D2,D3> const &v) {
      for (size_t i=0; i<totElems(); i++) { os << std::setw(12) << v.myData[i] << "\n"; }
      os << "\n";
      return os;
    }

    
    /** @brief Returns the dimensions of this array as a yakl::SArray object.
      * 
      * You should use zero-based indexing on the returned SArray object. */
    KOKKOS_INLINE_FUNCTION CSArray<size_t,1,rank> get_dimensions() const {
      CSArray<size_t,1,rank> ret;
      if constexpr (rank >= 1) ret(0) = D0;
      if constexpr (rank >= 2) ret(1) = D1;
      if constexpr (rank >= 3) ret(2) = D2;
      if constexpr (rank >= 4) ret(3) = D3;
      return ret;
    }
    /** @brief Returns the lower bound of each dimension of this array as a yakl::SArray object.
      * 
      * You should use zero-based indexing on the returned yakl::SArray object. */
    KOKKOS_INLINE_FUNCTION CSArray<size_t,1,rank> get_lbounds() const {
      CSArray<size_t,1,rank> ret;
      if constexpr (rank >= 1) ret(0) = 0;
      if constexpr (rank >= 2) ret(1) = 0;
      if constexpr (rank >= 3) ret(2) = 0;
      if constexpr (rank >= 4) ret(3) = 0;
      return ret;
    }
    /** @brief Returns the upper bound of each dimension of this array as a yakl::SArray object.
      *
      * You should use zero-based indexing on the returned yakl::SArray object. */
    KOKKOS_INLINE_FUNCTION CSArray<size_t,1,rank> get_ubounds() const {
      CSArray<size_t,1,rank> ret;
      if constexpr (rank >= 1) ret(0) = D0-1;
      if constexpr (rank >= 2) ret(1) = D1-1;
      if constexpr (rank >= 3) ret(2) = D2-1;
      if constexpr (rank >= 4) ret(3) = D3-1;
      return ret;
    }

  };



  /** @brief Most often, codes use the type define yakl::SArray rather than yakl::CSArray */
  template <class T, int rank, size_t D0, size_t D1=1, size_t D2=1, size_t D3=1>
  using SArray = CSArray<T,rank,D0,D1,D2,D3>;

}


