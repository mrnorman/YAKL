
#pragma once
// Included by YAKL_Array.h

namespace yakl {

  // [S]tatic (compile-time) Array [B]ounds (templated)
  // It's only used for Fortran, so it takes on Fortran defaults
  // with lower bound default to 1
  /** @brief This specifies a set of bounds for a dimension when declaring a yakl::FSArray.
    * 
    * It takes either one or two
    * template parameter. Specifying one template parameter gives an upper bound and assumes a lower bound of `1`. 
    * E.g., `yakl::SB<nx>` means a lower bound of `1` and an upper bound of `nx`. Specifying two template parameters
    * gives a lower and an upper bound. E.g., `yakl::SB<0,nx+1>` means a lower bound of `0` and an upper bound of `nx+1`.
    */
  template <int L, int U=-999> class SB {
  public:
    SB() = delete;
    static constexpr int lower() { return U == -999 ? 1 : L; }
    static constexpr int upper() { return U == -999 ? L : U; }
  };

  /*
    This is intended to be a simple, low-overhead class to do multi-dimensional arrays
    without pointer dereferencing. It supports indexing and cout only up to 4-D.
  */

  /** @brief Fortran-style array on the stack similar in nature to, e.g., `float arr[ny][nx];`
    *
    * This creates a Fortran-style "Stack Array" (FSArray) class. It should
    * be thought of as similar in nature to a C-style multi-dimensional array, `float arr[n1][n2][n3];`,
    * except that it uses column-majore index ordering (left-most index varies the fastest), and it has lower
    * bounds that default to `1` but can also be arbitrary.
    * An example of declaring a yakl:FSArray object is `yakl::FSArray<float,3,SB<n1>,SB<0,n2+1>,SB<n3>> arr;`
    * The syntax is a bit ugly, but it's necessary to allow lower bounds other than `1`. The array declared just
    * now will have lower bounds of 1, 0, and 1, respectively, and upper bounds of n1, n2+1, n3, respectively.
    * For bounds checking, define the CPP macro `KOKKOS_ENABLE_DEBUG`. Dimensions sizes must be
    * known at compile time, and data is placed on the stack of whatever context it is declared. When declared
    * in a device `parallel_for` kernel, it is a thread-private array, meaning every thread has a separate copy
    * of the array. 
    * @param T      Type of the yakl::FSArray object
    * @param rank   Number of dimensions
    * @param B[0-3] Bounds for each dimensions specified using a yakl::SB class. B1, B2, and B3 are optional template parameters.
    *               Each yakl::SB object can take one or two template parameters. Specifying only one template parameter assumes
    *               a lower bound of `1`. Specifying two template parameters gives the lower and upper bound (**inclusive**).
    * 
    * Creating these arrays is very cheap, but copying them does a deep copy every time and can be expensive.
    * Remember that yakl::FSArray objects use column-major index ordering, meaning the left-most index varies the fastest.
    */
  template <class T, int rank, class B0, class B1=SB<1>, class B2=SB<1>, class B3=SB<1>>
  class FSArray {
  protected:
    /** @private */
    static int constexpr U0 = B0::upper();
    /** @private */
    static int constexpr L0 = B0::lower();
    /** @private */
    static int constexpr U1 = B1::upper();
    /** @private */
    static int constexpr L1 = B1::lower();
    /** @private */
    static int constexpr U2 = B2::upper();
    /** @private */
    static int constexpr L2 = B2::lower();
    /** @private */
    static int constexpr U3 = B3::upper();
    /** @private */
    static int constexpr L3 = B3::lower();
    /** @private */
    static size_t constexpr D0 =             U0 - L0 + 1;
    /** @private */
    static size_t constexpr D1 = rank >= 1 ? U1 - L1 + 1 : 1;
    /** @private */
    static size_t constexpr D2 = rank >= 1 ? U2 - L2 + 1 : 1;
    /** @private */
    static size_t constexpr D3 = rank >= 1 ? U3 - L3 + 1 : 1;
    /** @private */
    static size_t constexpr OFF0 = 1;
    /** @private */
    static size_t constexpr OFF1 = D0;
    /** @private */
    static size_t constexpr OFF2 = D0*D1;
    /** @private */
    /** @private */
    T mutable myData[D0*D1*D2*D3];

  public :
    static size_t constexpr OFF3 = D0*D1*D2;

    /** @brief This is the type `T` without `const` and `volatile` modifiers */
    typedef typename std::remove_cv<T>::type       type;
    /** @brief This is the type `T` exactly as it was defined upon array object creation. */
    typedef          T                             value_type;
    /** @brief This is the type `T` with `const` added to it (if the original type has `volatile`, then so will this type). */
    typedef typename std::add_const<type>::type    const_value_type;
    /** @brief This is the type `T` with `const` removed from it (if the original type has `volatile`, then so will this type). */
    typedef typename std::remove_const<type>::type non_const_value_type;

    // All copies are deep, so be wary of copies. Use references where possible
    /** @brief No constructor arguments allowed */
    KOKKOS_INLINE_FUNCTION FSArray(T init_fill) { for (int i=0; i < size(); i++) { myData[i] = init_fill; } }
    FSArray()  = default;
    ~FSArray() = default;

    /** @brief Returns a reference to the indexed element (1-D).
      * @details Number of indices must match the rank of the array object. For bounds checking, define the CPP macro `KOKKOS_ENABLE_DEBUG`.
      * Always use one-based indexing (unless the dimension has non-default bounds) with column-major ordering (left-most index varying the fastest). */
    KOKKOS_INLINE_FUNCTION T &operator()(int const i0) const {
      static_assert(rank==1,"ERROR: Improper number of dimensions specified in operator()");
      #ifdef KOKKOS_ENABLE_DEBUG
        if constexpr (rank >= 1) { if (i0<L0 || i0>U0) { KOKKOS_IF_ON_HOST( printf("FSArray i0 out of bounds (i0: %d; lb0: %d; ub0: %d",i0,L0,U0); ) } }
        if constexpr (rank >= 1) { if (i0<L0 || i0>U0) { Kokkos::abort("ERROR: FSArray index out of bounds"); } }
      #endif
      return myData[i0-L0];
    }
    /** @brief Returns a reference to the indexed element (2-D).
      * @details Number of indices must match the rank of the array object. For bounds checking, define the CPP macro `KOKKOS_ENABLE_DEBUG`.
      * Always use one-based indexing (unless the dimension has non-default bounds) with column-major ordering (left-most index varying the fastest). */
    KOKKOS_INLINE_FUNCTION T &operator()(int const i0, int const i1) const {
      static_assert(rank==2,"ERROR: Improper number of dimensions specified in operator()");
      #ifdef KOKKOS_ENABLE_DEBUG
        if constexpr (rank >= 1) { if (i0<L0 || i0>U0) { KOKKOS_IF_ON_HOST( printf("FSArray i0 out of bounds (i0: %d; lb0: %d; ub0: %d",i0,L0,U0); ) } }
        if constexpr (rank >= 2) { if (i1<L1 || i1>U1) { KOKKOS_IF_ON_HOST( printf("FSArray i1 out of bounds (i1: %d; lb1: %d; ub1: %d",i1,L1,U1); ) } }
        if constexpr (rank >= 1) { if (i0<L0 || i0>U0) { Kokkos::abort("ERROR: FSArray index out of bounds"); } }
        if constexpr (rank >= 2) { if (i1<L1 || i1>U1) { Kokkos::abort("ERROR: FSArray index out of bounds"); } }
      #endif
      return myData[(i1-L1)*OFF1 + i0-L0];
    }
    /** @brief Returns a reference to the indexed element (3-D).
      * @details Number of indices must match the rank of the array object. For bounds checking, define the CPP macro `KOKKOS_ENABLE_DEBUG`.
      * Always use one-based indexing (unless the dimension has non-default bounds) with column-major ordering (left-most index varying the fastest). */
    KOKKOS_INLINE_FUNCTION T &operator()(int const i0, int const i1, int const i2) const {
      static_assert(rank==3,"ERROR: Improper number of dimensions specified in operator()");
      #ifdef KOKKOS_ENABLE_DEBUG
        if constexpr (rank >= 1) { if (i0<L0 || i0>U0) { KOKKOS_IF_ON_HOST( printf("FSArray i0 out of bounds (i0: %d; lb0: %d; ub0: %d",i0,L0,U0); ) } }
        if constexpr (rank >= 2) { if (i1<L1 || i1>U1) { KOKKOS_IF_ON_HOST( printf("FSArray i1 out of bounds (i1: %d; lb1: %d; ub1: %d",i1,L1,U1); ) } }
        if constexpr (rank >= 3) { if (i2<L2 || i2>U2) { KOKKOS_IF_ON_HOST( printf("FSArray i2 out of bounds (i2: %d; lb2: %d; ub2: %d",i2,L2,U2); ) } }
        if constexpr (rank >= 1) { if (i0<L0 || i0>U0) { Kokkos::abort("ERROR: FSArray index out of bounds"); } }
        if constexpr (rank >= 2) { if (i1<L1 || i1>U1) { Kokkos::abort("ERROR: FSArray index out of bounds"); } }
        if constexpr (rank >= 3) { if (i2<L2 || i2>U2) { Kokkos::abort("ERROR: FSArray index out of bounds"); } }
      #endif
      return myData[(i2-L2)*OFF2 + (i1-L1)*OFF1 + i0-L0];
    }
    /** @brief Returns a reference to the indexed element (4-D).
      * @details Number of indices must match the rank of the array object. For bounds checking, define the CPP macro `KOKKOS_ENABLE_DEBUG`.
      * Always use one-based indexing (unless the dimension has non-default bounds) with column-major ordering (left-most index varying the fastest). */
    KOKKOS_INLINE_FUNCTION T &operator()(int const i0, int const i1, int const i2, int const i3) const {
      static_assert(rank==4,"ERROR: Improper number of dimensions specified in operator()");
      #ifdef KOKKOS_ENABLE_DEBUG
        if constexpr (rank >= 1) { if (i0<L0 || i0>U0) { KOKKOS_IF_ON_HOST( printf("FSArray i0 out of bounds (i0: %d; lb0: %d; ub0: %d",i0,L0,U0); ) } }
        if constexpr (rank >= 2) { if (i1<L1 || i1>U1) { KOKKOS_IF_ON_HOST( printf("FSArray i1 out of bounds (i1: %d; lb1: %d; ub1: %d",i1,L1,U1); ) } }
        if constexpr (rank >= 3) { if (i2<L2 || i2>U2) { KOKKOS_IF_ON_HOST( printf("FSArray i2 out of bounds (i2: %d; lb2: %d; ub2: %d",i2,L2,U2); ) } }
        if constexpr (rank >= 4) { if (i3<L3 || i3>U3) { KOKKOS_IF_ON_HOST( printf("FSArray i3 out of bounds (i3: %d; lb3: %d; ub3: %d",i3,L3,U3); ) } }
        if constexpr (rank >= 1) { if (i0<L0 || i0>U0) { Kokkos::abort("ERROR: FSArray index out of bounds"); } }
        if constexpr (rank >= 2) { if (i1<L1 || i1>U1) { Kokkos::abort("ERROR: FSArray index out of bounds"); } }
        if constexpr (rank >= 3) { if (i2<L2 || i2>U2) { Kokkos::abort("ERROR: FSArray index out of bounds"); } }
        if constexpr (rank >= 4) { if (i3<L3 || i3>U3) { Kokkos::abort("ERROR: FSArray index out of bounds"); } }
      #endif
      return myData[(i3-L3)*OFF3 + (i2-L2)*OFF2 + (i1-L1)*OFF1 + i0-L0];
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
    static size_t constexpr get_elem_count() { return D3*D2*D1*D0; }
    /** @brief Get the total number of array elements */
    static size_t constexpr size          () { return D3*D2*D1*D0; }
    /** @brief Get the number of dimensions */
    static size_t constexpr get_rank      () { return rank; }
    /** @brief Always true. All YAKL arrays are contiguous with no padding. */
    static bool     constexpr span_is_contiguous() { return true; }
    /** @brief Always true. yakl::SArray objects are by default always initialized / allocated. */
    static bool     constexpr initialized() { return true; }


    /** @brief Print out the contents of this array. This should be called only from the host */
    inline friend std::ostream &operator<<(std::ostream& os, FSArray const &v) {
      for (int i=0; i<totElems(); i++) { os << std::setw(12) << v.myData[i] << "\n"; }
      os << "\n";
      return os;
    }

    
    /** @brief Returns the dimensions of this array as a yakl::FSArray object.
      * 
      * You should use one-based indexing on the returned yakl::FSArray object. */
    KOKKOS_INLINE_FUNCTION FSArray<int,1,SB<rank>> get_dimensions() const {
      FSArray<int,1,SB<rank>> ret;
      if constexpr (rank >= 1) ret(1) = D0;
      if constexpr (rank >= 2) ret(2) = D1;
      if constexpr (rank >= 3) ret(3) = D2;
      if constexpr (rank >= 4) ret(4) = D3;
      return ret;
    }
    /** @brief Returns the lower bound of each dimension of this array as a yakl::FSArray object.
      * 
      * You should use one-based indexing on the returned yakl::FSArray object. */
    KOKKOS_INLINE_FUNCTION FSArray<int,1,SB<rank>> get_lbounds() const {
      FSArray<int,1,SB<rank>> ret;
      if constexpr (rank >= 1) ret(1) = L0;
      if constexpr (rank >= 2) ret(2) = L1;
      if constexpr (rank >= 3) ret(3) = L2;
      if constexpr (rank >= 4) ret(4) = L3;
      return ret;
    }
    /** @brief Returns the upper bound of each dimension of this array as a yakl::FSArray object.
      * 
      * You should use one-based indexing on the returned yakl::FSArray object. */
    KOKKOS_INLINE_FUNCTION FSArray<int,1,SB<rank>> get_ubounds() const {
      FSArray<int,1,SB<rank>> ret;
      if constexpr (rank >= 1) ret(1) = U0;
      if constexpr (rank >= 2) ret(2) = U1;
      if constexpr (rank >= 3) ret(3) = U2;
      if constexpr (rank >= 4) ret(4) = U3;
      return ret;
    }

  };

}


