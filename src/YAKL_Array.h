/**
 * @file
 *
 * Declare the yakl::Array object template and defaults. Declares yakl::Dim, yakl::Dims, yakl::Bnd, and yakl::Bnds
 * classes.
 */

#pragma once
// Included by YAKL.h

#include "YAKL_CSArray.h"
#include "YAKL_FSArray.h"

namespace yakl {

  // Labels for Array styles. C has zero-based indexing with the last index varying the fastest.
  // Fortran has 1-based indexing with arbitrary lower bounds and the index varying the fastest.
  /** @brief Template parameter for yakl::Array that specifies it should follow C-style behavior */
  int constexpr styleC       = 1;
  /** @brief Template parameter for yakl::Array that specifies it should follow Fortran-style behavior */
  int constexpr styleFortran = 2;
  /** @brief Default style is C-style for yakl::Array objects */
  int constexpr styleDefault = styleC;

  /** @brief This is just a convenience syntax for slicing yakl::Array objects to make it clear in the user
             level code which dimensions are being sliced. */
  int constexpr COLON = std::numeric_limits<int>::min(); // Label for the ":" from Fortrna array slicing

  // The one template to rule them all for the Array class
  // This ultimately describes dynamics and static / stack Arrays in all types, ranks, memory spaces, and styles
  /** @brief This declares the yakl::Array class. Please see the yakl::styleC and yakl::styleFortran template
    *        specializations for more detailed information about this class.
    * @param T       The type of the array. For yakl::memHost arrays, this can be generally any time. For yakl::memDevice
    *                arrays, this needs to be a type with no constructor, preferably an arithmetic type.
    * @param rank    The number of dimensions for this array object.
    * @param myMem   The memory space for this array object: either yakl::memHost or yakl::memDevice
    * @param myStyle The behavior of this array: either yakl::styleC or yakl::styleFortran
    */
  template <class T, int rank, int myMem=memDefault, int myStyle=styleDefault> class Array;


  // This class is used to describe a set of dimensions used for Array slicing. One can call a constructor
  // with std::initialize_list (i.e., {1,2,3...} syntax)
  /** @brief This class holds C-style dimensions for using in yakl::Array objects.
    * 
    * You can pass an initializer list `{...}`  or std::vector as a parameter to this type,
    * and it can be converted to a yakl::Dims object.
    */
  class Dims {
  public:
    /** @private */
    int data[8];
    /** @private */
    int rank;

    YAKL_INLINE Dims() {rank = 0;}
    /** @brief Construct a 1-D Dims object) */
    YAKL_INLINE Dims(int i0) {
      data[0] = i0;
      rank = 1;
    }
    /** @brief Construct a 2-D Dims object) */
    YAKL_INLINE Dims(int i0, int i1) {
      data[0] = i0;
      data[1] = i1;
      rank = 2;
    }
    /** @brief Construct a 3-D Dims object) */
    YAKL_INLINE Dims(int i0, int i1, int i2) {
      data[0] = i0;
      data[1] = i1;
      data[2] = i2;
      rank = 3;
    }
    /** @brief Construct a 4-D Dims object) */
    YAKL_INLINE Dims(int i0, int i1, int i2, int i3) {
      data[0] = i0;
      data[1] = i1;
      data[2] = i2;
      data[3] = i3;
      rank = 4;
    }
    /** @brief Construct a 5-D Dims object) */
    YAKL_INLINE Dims(int i0, int i1, int i2, int i3, int i4) {
      data[0] = i0;
      data[1] = i1;
      data[2] = i2;
      data[3] = i3;
      data[4] = i4;
      rank = 5;
    }
    /** @brief Construct a 6-D Dims object) */
    YAKL_INLINE Dims(int i0, int i1, int i2, int i3, int i4, int i5) {
      data[0] = i0;
      data[1] = i1;
      data[2] = i2;
      data[3] = i3;
      data[4] = i4;
      data[5] = i5;
      rank = 6;
    }
    /** @brief Construct a 7-D Dims object) */
    YAKL_INLINE Dims(int i0, int i1, int i2, int i3, int i4, int i5, int i6) {
      data[0] = i0;
      data[1] = i1;
      data[2] = i2;
      data[3] = i3;
      data[4] = i4;
      data[5] = i5;
      data[6] = i6;
      rank = 7;
    }
    /** @brief Construct an 8-D Dims object) */
    YAKL_INLINE Dims(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7) {
      data[0] = i0;
      data[1] = i1;
      data[2] = i2;
      data[3] = i3;
      data[4] = i4;
      data[5] = i5;
      data[6] = i6;
      data[7] = i7;
      rank = 8;
    }

    YAKL_INLINE Dims(Dims const &dims) {
      rank = dims.rank;
      for (int i=0; i < rank; i++) { data[i] = dims[i]; }
    }
    YAKL_INLINE Dims &operator=(Dims const &dims) {
      if (this == &dims) { return *this; }
      rank = dims.rank;
      for (int i=0; i < rank; i++) { data[i] = dims[i]; }
      return *this;
    }
    YAKL_INLINE Dims(Dims &&dims) {
      rank = dims.rank;
      for (int i=0; i < rank; i++) { data[i] = dims[i]; }
    }
    YAKL_INLINE Dims &operator=(Dims &&dims) {
      if (this == &dims) { return *this; }
      rank = dims.rank;
      for (int i=0; i < rank; i++) { data[i] = dims[i]; }
      return *this;
    }

    /** @brief This constructor allows converting a std::vector or initializer list to a yakl::Dims object. */
    template <class INT, typename std::enable_if< std::is_integral<INT>::value , bool>::type = false>
    Dims(std::vector<INT> const dims) {
      rank = dims.size();
      for (int i=0; i < rank; i++) { data[i] = dims[i]; }
    }

    /** @brief This constructor allows converting a CSArray object to a yakl::Dims object. */
    template <class INT, unsigned int RANK, typename std::enable_if< std::is_integral<INT>::value , bool>::type = false>
    Dims(CSArray<INT,1,RANK> const dims) {
      rank = RANK;
      for (int i=0; i < rank; i++) { data[i] = dims(i); }
    }

    /** @brief These objects are always indexed with square bracket notation like a std::vector or std::array. */
    YAKL_INLINE int operator[] (int i) const { return data[i]; }

    /** @brief Get the number of dimensions. */
    YAKL_INLINE int size() const { return rank; }
  };



  // Describes a single array bound. Used for Fortran array bounds
  /** @brief Describes a single bound for creating Fortran-style yakl::Array objects.
    * 
    * You can create and object of this class with a single integer, in which case the lower bound default to one,
    * or you can use a pair of values, e.g., `{lower,upper}`, which will assign these as lower and upper bounds.
    */
  struct Bnd {
  public:
    /** @private */
    int l;
    /** @private */
    int u;
    /** @brief Create a dimension bound with a lower limit of 1 and an upper limit of 1 */
    YAKL_INLINE Bnd(                  ) { l = 1   ; u = 1   ; }
    /** @brief Create a dimension bound with a lower limit of 1 and an upper limit of u_in */
    YAKL_INLINE Bnd(          int u_in) { l = 1   ; u = u_in; }
    /** @brief Create a dimension bound with a lower limit of l_in and an upper limit of u_in */
    YAKL_INLINE Bnd(int l_in, int u_in) { l = l_in; u = u_in; }
  };



  // Describes a set of array bounds. use for Fortran array bounds
  /** @brief This class holds Fortran-style dimensions for using in creating yakl::Array objects.
    * 
    * You can pass an initializer list `{...}` or std::vector as a parameter to this type, and it can be converted
    * to a yakl::Bnds object. Each `Bnd` element you pass can be an integer upper bound value (lower bound defaults
    * to one) or a pair (`{lower,upper}`), since Fortran-style allows arbitrary lower bounds. 
    */
  class Bnds {
  public:
    /** @private */
    int l[8];
    /** @private */
    int u[8];
    /** @private */
    int rank;

    YAKL_INLINE Bnds() {rank = 0;}
    /** @brief Construct an 1-D Dims object) */
    YAKL_INLINE Bnds(Bnd b0) {
      l[0] = b0.l;   u[0] = b0.u;
      rank = 1;
    }
    /** @brief Construct an 2-D Dims object) */
    YAKL_INLINE Bnds(Bnd b0, Bnd b1) {
      l[0] = b0.l;   u[0] = b0.u;
      l[1] = b1.l;   u[1] = b1.u;
      rank = 2;
    }
    /** @brief Construct an 3-D Dims object) */
    YAKL_INLINE Bnds(Bnd b0, Bnd b1, Bnd b2) {
      l[0] = b0.l;   u[0] = b0.u;
      l[1] = b1.l;   u[1] = b1.u;
      l[2] = b2.l;   u[2] = b2.u;
      rank = 3;
    }
    /** @brief Construct an 4-D Dims object) */
    YAKL_INLINE Bnds(Bnd b0, Bnd b1, Bnd b2, Bnd b3) {
      l[0] = b0.l;   u[0] = b0.u;
      l[1] = b1.l;   u[1] = b1.u;
      l[2] = b2.l;   u[2] = b2.u;
      l[3] = b3.l;   u[3] = b3.u;
      rank = 4;
    }
    /** @brief Construct an 5-D Dims object) */
    YAKL_INLINE Bnds(Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4) {
      l[0] = b0.l;   u[0] = b0.u;
      l[1] = b1.l;   u[1] = b1.u;
      l[2] = b2.l;   u[2] = b2.u;
      l[3] = b3.l;   u[3] = b3.u;
      l[4] = b4.l;   u[4] = b4.u;
      rank = 5;
    }
    /** @brief Construct an 6-D Dims object) */
    YAKL_INLINE Bnds(Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5) {
      l[0] = b0.l;   u[0] = b0.u;
      l[1] = b1.l;   u[1] = b1.u;
      l[2] = b2.l;   u[2] = b2.u;
      l[3] = b3.l;   u[3] = b3.u;
      l[4] = b4.l;   u[4] = b4.u;
      l[5] = b5.l;   u[5] = b5.u;
      rank = 6;
    }
    /** @brief Construct an 7-D Dims object) */
    YAKL_INLINE Bnds(Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5, Bnd b6) {
      l[0] = b0.l;    u[0] = b0.u;
      l[1] = b1.l;    u[1] = b1.u;
      l[2] = b2.l;    u[2] = b2.u;
      l[3] = b3.l;    u[3] = b3.u;
      l[4] = b4.l;    u[4] = b4.u;
      l[5] = b5.l;    u[5] = b5.u;
      l[6] = b6.l;    u[6] = b6.u;
      rank = 7;
    }
    /** @brief Construct an 8-D Dims object) */
    YAKL_INLINE Bnds(Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5, Bnd b6, Bnd b7) {
      l[0] = b0.l;   u[0] = b0.u;
      l[1] = b1.l;   u[1] = b1.u;
      l[2] = b2.l;   u[2] = b2.u;
      l[3] = b3.l;   u[3] = b3.u;
      l[4] = b4.l;   u[4] = b4.u;
      l[5] = b5.l;   u[5] = b5.u;
      l[6] = b6.l;   u[6] = b6.u;
      l[7] = b7.l;   u[7] = b7.u;
      rank = 8;
    }
    YAKL_INLINE Bnds(Bnds const &bnds) {
      rank = bnds.rank;
      for (int i=0; i < rank; i++) { l[i] = bnds.l[i]; u[i] = bnds.u[i]; }
    }
    YAKL_INLINE Bnds &operator=(Bnds const &bnds) {
      if (this == &bnds) { return *this; }
      rank = bnds.rank;
      for (int i=0; i < rank; i++) { l[i] = bnds.l[i]; u[i] = bnds.u[i]; }
      return *this;
    }
    YAKL_INLINE Bnds(Bnds &&bnds) {
      rank = bnds.rank;
      for (int i=0; i < rank; i++) { l[i] = bnds.l[i]; u[i] = bnds.u[i]; }
    }
    YAKL_INLINE Bnds &operator=(Bnds &&bnds) {
      if (this == &bnds) { return *this; }
      rank = bnds.rank;
      for (int i=0; i < rank; i++) { l[i] = bnds.l[i]; u[i] = bnds.u[i]; }
      return *this;
    }
    /** @brief Allows an initializer list or std::vector to be converted to a yakl::Bnds object */
    Bnds(std::vector<Bnd> const bnds) {
      rank = bnds.size();
      for (int i=0; i < rank; i++) { l[i] = bnds[i].l;   u[i] = bnds[i].u; }
    }
    /** @brief Allows an initializer list or std::vector to be converted to a yakl::Bnds object */
    template <class INT, typename std::enable_if< std::is_integral<INT>::value , bool>::type = false>
    Bnds(std::vector<INT> const bnds) {
      rank = bnds.size();
      for (int i=0; i < rank; i++) { l[i] = 1;   u[i] = bnds[i]; }
    }

    /** @brief Allows CSArray object to be converted to a yakl::Bnds object if default lower bounds of 1 are desired for
      *        all dimensions. */
    template <class INT, unsigned int RANK, typename std::enable_if< std::is_integral<INT>::value , bool>::type = false>
    Bnds(CSArray<INT,1,RANK> const dims) {
      rank = RANK;
      for (int i=0; i < rank; i++) { l[i] = 1;   u[i] = dims(i); }
    }

    /** @brief Allows FSArray object to be converted to a yakl::Bnds object if default lower bounds of 1 are desired for
      *        all dimensions. */
    template <class INT, int LOWER, int UPPER, typename std::enable_if< std::is_integral<INT>::value , bool>::type = false>
    Bnds(FSArray<INT,1,SB<LOWER,UPPER>> const dims) {
      rank = UPPER-LOWER+1;
      for (int i=LOWER; i <= UPPER; i++) { l[i] = 1;   u[i] = dims(i); }
    }

    /** @brief Allows two FSArray objects (one for lower bounds, one for upper bounds) to be converted to a yakl::Bnds
      * object. */
    template <class INT, int LOWER1, int UPPER1, int LOWER2, int UPPER2, typename std::enable_if< std::is_integral<INT>::value , bool>::type = false>
    Bnds(FSArray<INT,1,SB<LOWER1,UPPER1>> const lbounds, FSArray<INT,1,SB<LOWER2,UPPER2>> const ubounds) {
      static_assert( UPPER1-LOWER1+1 == UPPER2-LOWER2+1 , "ERROR: lbounds and ubounds sizes are not equal" );
      rank = UPPER1-LOWER1+1;
      for (int i=LOWER1; i <= UPPER1; i++) { l[i] = lbounds(i); }
      for (int i=LOWER2; i <= UPPER2; i++) { u[i] = ubounds(i); }
    }

    /** @brief These objects are always indexed with square bracket notation like a std::vector or std::array. */
    YAKL_INLINE Bnd operator[] (int i) const { return Bnd(l[i],u[i]); }

    /** @brief Get the number of dimensions. */
    YAKL_INLINE int size() const { return rank; }
  };

}

#include "YAKL_ArrayBase.h"
#include "YAKL_CArray.h"
#include "YAKL_FArray.h"


