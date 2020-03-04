
#pragma once

template <class T, int rank, int myMem> class Array<T,rank,myMem,styleFortran> {

  public :

  size_t offsets  [rank];  // Precomputed dimension offsets for efficient data access into a 1-D pointer
  int    lbounds  [rank];  // Lower bounds for each dimension
  int    dimension[rank];  // Sizes of dimensions
  T      * myData;         // Pointer to the flattened internal data
  int    * refCount;       // Pointer shared by multiple copies of this Array to keep track of allcation / free
  bool   owned;            // Whether is is owned (owned = allocated,ref_counted,deallocated) or not
  #ifdef ARRAY_DEBUG
    std::string myname; // Label for debug printing. Only stored if debugging is turned on
  #endif


  // Start off all constructors making sure the pointers are null
  YAKL_INLINE void nullify() {
    owned    = true;
    myData   = nullptr;
    refCount = nullptr;
  }

  /* CONSTRUCTORS
  You can declare the array empty or with up to 8 dimensions
  Like kokkos, you need to give a label for the array for debug printing
  Always nullify before beginning so that myData == nullptr upon init. This allows the
  setup() functions to keep from deallocating myData upon initialization, since
  you don't know what "myData" will be when the object is created.
  */
  Array() {
    nullify();
  }
  Array(char const * label) {
    nullify();
    #ifdef ARRAY_DEBUG
      myname = std::string(label);
    #endif
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Owned constructors
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  //Define the dimension ranges using an array of upper bounds, assuming lower bounds to be one
  Array(char const * label, int const d1) {
    static_assert( rank == 1 , "ERROR: Calling invalid constructor on rank 1 Array" );
    nullify();
    for (int i=0; i<1; i++) { lbounds[i] = 1; }
    setup(label,d1);
  }
  Array(char const * label, int const d1, int const d2) {
    static_assert( rank == 2 , "ERROR: Calling invalid constructor on rank 2 Array" );
    nullify();
    for (int i=0; i<2; i++) { lbounds[i] = 1; }
    setup(label,d1,d2);
  }
  Array(char const * label, int const d1, int const d2, int const d3) {
    static_assert( rank == 3 , "ERROR: Calling invalid constructor on rank 3 Array" );
    nullify();
    for (int i=0; i<3; i++) { lbounds[i] = 1; }
    setup(label,d1,d2,d3);
  }
  Array(char const * label, int const d1, int const d2, int const d3, int const d4) {
    static_assert( rank == 4 , "ERROR: Calling invalid constructor on rank 4 Array" );
    nullify();
    for (int i=0; i<4; i++) { lbounds[i] = 1; }
    setup(label,d1,d2,d3,d4);
  }
  Array(char const * label, int const d1, int const d2, int const d3, int const d4, int const d5) {
    static_assert( rank == 5 , "ERROR: Calling invalid constructor on rank 5 Array" );
    nullify();
    for (int i=0; i<5; i++) { lbounds[i] = 1; }
    setup(label,d1,d2,d3,d4,d5);
  }
  Array(char const * label, int const d1, int const d2, int const d3, int const d4, int const d5, int const d6) {
    static_assert( rank == 6 , "ERROR: Calling invalid constructor on rank 6 Array" );
    nullify();
    for (int i=0; i<6; i++) { lbounds[i] = 1; }
    setup(label,d1,d2,d3,d4,d5,d6);
  }
  Array(char const * label, int const d1, int const d2, int const d3, int const d4, int const d5, int const d6, int const d7) {
    static_assert( rank == 7 , "ERROR: Calling invalid constructor on rank 7 Array" );
    nullify();
    for (int i=0; i<7; i++) { lbounds[i] = 1; }
    setup(label,d1,d2,d3,d4,d5,d6,d7);
  }
  Array(char const * label, int const d1, int const d2, int const d3, int const d4, int const d5, int const d6, int const d7, int const d8) {
    static_assert( rank == 8 , "ERROR: Calling invalid constructor on rank 8 Array" );
    nullify();
    for (int i=0; i<8; i++) { lbounds[i] = 1; }
    setup(label,d1,d2,d3,d4,d5,d6,d7,d8);
  }
  
  //Define the dimension ranges using bounding pairs of {lowerbound,higherbound}
  Array(char const * label, Bnd const &b1) {
    static_assert( rank == 1 , "ERROR: Calling invalid constructor on rank 1 Array" );
    nullify();
    lbounds[0] = b1.l;
    int d1 = b1.u - b1.l + 1;
    setup(label,d1);
  }
  Array(char const * label, Bnd const &b1, Bnd const &b2) {
    static_assert( rank == 2 , "ERROR: Calling invalid constructor on rank 2 Array" );
    nullify();
    lbounds[0] = b1.l;
    lbounds[1] = b2.l;
    int d1 = b1.u - b1.l + 1;
    int d2 = b2.u - b2.l + 1;
    setup(label,d1,d2);
  }
  Array(char const * label, Bnd const &b1, Bnd const &b2, Bnd const &b3) {
    static_assert( rank == 3 , "ERROR: Calling invalid constructor on rank 3 Array" );
    nullify();
    lbounds[0] = b1.l;
    lbounds[1] = b2.l;
    lbounds[2] = b3.l;
    int d1 = b1.u - b1.l + 1;
    int d2 = b2.u - b2.l + 1;
    int d3 = b3.u - b3.l + 1;
    setup(label,d1,d2,d3);
  }
  Array(char const * label, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4) {
    static_assert( rank == 4 , "ERROR: Calling invalid constructor on rank 4 Array" );
    nullify();
    lbounds[0] = b1.l;
    lbounds[1] = b2.l;
    lbounds[2] = b3.l;
    lbounds[3] = b4.l;
    int d1 = b1.u - b1.l + 1;
    int d2 = b2.u - b2.l + 1;
    int d3 = b3.u - b3.l + 1;
    int d4 = b4.u - b4.l + 1;
    setup(label,d1,d2,d3,d4);
  }
  Array(char const * label, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5) {
    static_assert( rank == 5 , "ERROR: Calling invalid constructor on rank 5 Array" );
    nullify();
    lbounds[0] = b1.l;
    lbounds[1] = b2.l;
    lbounds[2] = b3.l;
    lbounds[3] = b4.l;
    lbounds[4] = b5.l;
    int d1 = b1.u - b1.l + 1;
    int d2 = b2.u - b2.l + 1;
    int d3 = b3.u - b3.l + 1;
    int d4 = b4.u - b4.l + 1;
    int d5 = b5.u - b5.l + 1;
    setup(label,d1,d2,d3,d4,d5);
  }
  Array(char const * label, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5, Bnd const &b6) {
    static_assert( rank == 6 , "ERROR: Calling invalid constructor on rank 6 Array" );
    nullify();
    lbounds[0] = b1.l;
    lbounds[1] = b2.l;
    lbounds[2] = b3.l;
    lbounds[3] = b4.l;
    lbounds[4] = b5.l;
    lbounds[5] = b6.l;
    int d1 = b1.u - b1.l + 1;
    int d2 = b2.u - b2.l + 1;
    int d3 = b3.u - b3.l + 1;
    int d4 = b4.u - b4.l + 1;
    int d5 = b5.u - b5.l + 1;
    int d6 = b6.u - b6.l + 1;
    setup(label,d1,d2,d3,d4,d5,d6);
  }
  Array(char const * label, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5, Bnd const &b6, Bnd const &b7) {
    static_assert( rank == 7 , "ERROR: Calling invalid constructor on rank 7 Array" );
    nullify();
    lbounds[0] = b1.l;
    lbounds[1] = b2.l;
    lbounds[2] = b3.l;
    lbounds[3] = b4.l;
    lbounds[4] = b5.l;
    lbounds[5] = b6.l;
    lbounds[6] = b7.l;
    int d1 = b1.u - b1.l + 1;
    int d2 = b2.u - b2.l + 1;
    int d3 = b3.u - b3.l + 1;
    int d4 = b4.u - b4.l + 1;
    int d5 = b5.u - b5.l + 1;
    int d6 = b6.u - b6.l + 1;
    int d7 = b7.u - b7.l + 1;
    setup(label,d1,d2,d3,d4,d5,d6,d7);
  }
  Array(char const * label, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5, Bnd const &b6, Bnd const &b7, Bnd const &b8) {
    static_assert( rank == 8 , "ERROR: Calling invalid constructor on rank 8 Array" );
    nullify();
    lbounds[0] = b1.l;
    lbounds[1] = b2.l;
    lbounds[2] = b3.l;
    lbounds[3] = b4.l;
    lbounds[4] = b5.l;
    lbounds[5] = b6.l;
    lbounds[6] = b7.l;
    lbounds[7] = b8.l;
    int d1 = b1.u - b1.l + 1;
    int d2 = b2.u - b2.l + 1;
    int d3 = b3.u - b3.l + 1;
    int d4 = b4.u - b4.l + 1;
    int d5 = b5.u - b5.l + 1;
    int d6 = b6.u - b6.l + 1;
    int d7 = b7.u - b7.l + 1;
    int d8 = b8.u - b8.l + 1;
    setup(label,d1,d2,d3,d4,d5,d6,d7,d8);
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Non-owned constructors
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Define the dimension ranges using an array of upper bounds, assuming lower bounds to be zero
  Array(char const * label, T * data, int const d1) {
    static_assert( rank == 1 , "ERROR: Calling invalid constructor on rank 1 Array" );
    nullify();
    owned = false;
    for (int i=0; i<1; i++) { lbounds[i] = 1; }
    setup(label,d1);
    myData = data;
  }
  Array(char const * label, T * data, int const d1, int const d2) {
    static_assert( rank == 2 , "ERROR: Calling invalid constructor on rank 2 Array" );
    nullify();
    owned = false;
    for (int i=0; i<2; i++) { lbounds[i] = 1; }
    setup(label,d1,d2);
    myData = data;
  }
  Array(char const * label, T * data, int const d1, int const d2, int const d3) {
    static_assert( rank == 3 , "ERROR: Calling invalid constructor on rank 3 Array" );
    nullify();
    owned = false;
    for (int i=0; i<3; i++) { lbounds[i] = 1; }
    setup(label,d1,d2,d3);
    myData = data;
  }
  Array(char const * label, T * data, int const d1, int const d2, int const d3, int const d4) {
    static_assert( rank == 4 , "ERROR: Calling invalid constructor on rank 4 Array" );
    nullify();
    owned = false;
    for (int i=0; i<4; i++) { lbounds[i] = 1; }
    setup(label,d1,d2,d3,d4);
    myData = data;
  }
  Array(char const * label, T * data, int const d1, int const d2, int const d3, int const d4, int const d5) {
    static_assert( rank == 5 , "ERROR: Calling invalid constructor on rank 5 Array" );
    nullify();
    owned = false;
    for (int i=0; i<5; i++) { lbounds[i] = 1; }
    setup(label,d1,d2,d3,d4,d5);
    myData = data;
  }
  Array(char const * label, T * data, int const d1, int const d2, int const d3, int const d4, int const d5, int const d6) {
    static_assert( rank == 6 , "ERROR: Calling invalid constructor on rank 6 Array" );
    nullify();
    owned = false;
    for (int i=0; i<6; i++) { lbounds[i] = 1; }
    setup(label,d1,d2,d3,d4,d5,d6);
    myData = data;
  }
  Array(char const * label, T * data, int const d1, int const d2, int const d3, int const d4, int const d5, int const d6, int const d7) {
    static_assert( rank == 7 , "ERROR: Calling invalid constructor on rank 7 Array" );
    nullify();
    owned = false;
    for (int i=0; i<7; i++) { lbounds[i] = 1; }
    setup(label,d1,d2,d3,d4,d5,d6,d7);
    myData = data;
  }
  Array(char const * label, T * data, int const d1, int const d2, int const d3, int const d4, int const d5, int const d6, int const d7, int const d8) {
    static_assert( rank == 8 , "ERROR: Calling invalid constructor on rank 8 Array" );
    nullify();
    owned = false;
    for (int i=0; i<8; i++) { lbounds[i] = 1; }
    setup(label,d1,d2,d3,d4,d5,d6,d7,d8);
    myData = data;
  }

  // Define the dimension ranges using bounding pairs of {lowerbound,higherbound}
  Array(char const * label, T * data, Bnd const &b1) {
    static_assert( rank == 1 , "ERROR: Calling invalid constructor on rank 1 Array" );
    nullify();
    owned = false;
    lbounds[0] = b1.l;
    int d1 = b1.u - b1.l + 1;
    setup(label,d1);
    myData = data;
  }
  Array(char const * label, T * data, Bnd const &b1, Bnd const &b2) {
    static_assert( rank == 2 , "ERROR: Calling invalid constructor on rank 2 Array" );
    nullify();
    owned = false;
    lbounds[0] = b1.l;
    lbounds[1] = b2.l;
    int d1 = b1.u - b1.l + 1;
    int d2 = b2.u - b2.l + 1;
    setup(label,d1,d2);
    myData = data;
  }
  Array(char const * label, T * data, Bnd const &b1, Bnd const &b2, Bnd const &b3) {
    static_assert( rank == 3 , "ERROR: Calling invalid constructor on rank 3 Array" );
    nullify();
    owned = false;
    lbounds[0] = b1.l;
    lbounds[1] = b2.l;
    lbounds[2] = b3.l;
    int d1 = b1.u - b1.l + 1;
    int d2 = b2.u - b2.l + 1;
    int d3 = b3.u - b3.l + 1;
    setup(label,d1,d2,d3);
    myData = data;
  }
  Array(char const * label, T * data, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4) {
    static_assert( rank == 4 , "ERROR: Calling invalid constructor on rank 4 Array" );
    nullify();
    owned = false;
    lbounds[0] = b1.l;
    lbounds[1] = b2.l;
    lbounds[2] = b3.l;
    lbounds[3] = b4.l;
    int d1 = b1.u - b1.l + 1;
    int d2 = b2.u - b2.l + 1;
    int d3 = b3.u - b3.l + 1;
    int d4 = b4.u - b4.l + 1;
    setup(label,d1,d2,d3,d4);
    myData = data;
  }
  Array(char const * label, T * data, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5) {
    static_assert( rank == 5 , "ERROR: Calling invalid constructor on rank 5 Array" );
    nullify();
    owned = false;
    lbounds[0] = b1.l;
    lbounds[1] = b2.l;
    lbounds[2] = b3.l;
    lbounds[3] = b4.l;
    lbounds[4] = b5.l;
    int d1 = b1.u - b1.l + 1;
    int d2 = b2.u - b2.l + 1;
    int d3 = b3.u - b3.l + 1;
    int d4 = b4.u - b4.l + 1;
    int d5 = b5.u - b5.l + 1;
    setup(label,d1,d2,d3,d4,d5);
    myData = data;
  }
  Array(char const * label, T * data, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5, Bnd const &b6) {
    static_assert( rank == 6 , "ERROR: Calling invalid constructor on rank 6 Array" );
    nullify();
    owned = false;
    lbounds[0] = b1.l;
    lbounds[1] = b2.l;
    lbounds[2] = b3.l;
    lbounds[3] = b4.l;
    lbounds[4] = b5.l;
    lbounds[5] = b6.l;
    int d1 = b1.u - b1.l + 1;
    int d2 = b2.u - b2.l + 1;
    int d3 = b3.u - b3.l + 1;
    int d4 = b4.u - b4.l + 1;
    int d5 = b5.u - b5.l + 1;
    int d6 = b6.u - b6.l + 1;
    setup(label,d1,d2,d3,d4,d5,d6);
    myData = data;
  }
  Array(char const * label, T * data, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5, Bnd const &b6, Bnd const &b7) {
    static_assert( rank == 7 , "ERROR: Calling invalid constructor on rank 7 Array" );
    nullify();
    owned = false;
    lbounds[0] = b1.l;
    lbounds[1] = b2.l;
    lbounds[2] = b3.l;
    lbounds[3] = b4.l;
    lbounds[4] = b5.l;
    lbounds[5] = b6.l;
    lbounds[6] = b7.l;
    int d1 = b1.u - b1.l + 1;
    int d2 = b2.u - b2.l + 1;
    int d3 = b3.u - b3.l + 1;
    int d4 = b4.u - b4.l + 1;
    int d5 = b5.u - b5.l + 1;
    int d6 = b6.u - b6.l + 1;
    int d7 = b7.u - b7.l + 1;
    setup(label,d1,d2,d3,d4,d5,d6,d7);
    myData = data;
  }
  Array(char const * label, T * data, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5, Bnd const &b6, Bnd const &b7, Bnd const &b8) {
    static_assert( rank == 8 , "ERROR: Calling invalid constructor on rank 8 Array" );
    nullify();
    owned = false;
    lbounds[0] = b1.l;
    lbounds[1] = b2.l;
    lbounds[2] = b3.l;
    lbounds[3] = b4.l;
    lbounds[4] = b5.l;
    lbounds[5] = b6.l;
    lbounds[6] = b7.l;
    lbounds[7] = b8.l;
    int d1 = b1.u - b1.l + 1;
    int d2 = b2.u - b2.l + 1;
    int d3 = b3.u - b3.l + 1;
    int d4 = b4.u - b4.l + 1;
    int d5 = b5.u - b5.l + 1;
    int d6 = b6.u - b6.l + 1;
    int d7 = b7.u - b7.l + 1;
    int d8 = b8.u - b8.l + 1;
    setup(label,d1,d2,d3,d4,d5,d6,d7,d8);
    myData = data;
  }


  /*
  COPY CONSTRUCTORS / FUNCTIONS
  This shares the pointers with another Array and increments the refCounter
  */
  Array(Array const &rhs) {
    nullify();
    owned    = rhs.owned;
    for (int i=0; i<rank; i++) {
      offsets  [i] = rhs.offsets  [i];
      lbounds  [i] = rhs.lbounds  [i];
      dimension[i] = rhs.dimension[i];
    }
    #ifdef ARRAY_DEBUG
      myname = rhs.myname;
    #endif
    myData   = rhs.myData;
    refCount = rhs.refCount;
    if (owned) { (*refCount)++; }
  }


  Array & operator=(Array const &rhs) {
    if (this == &rhs) {
      return *this;
    }
    owned    = rhs.owned;
    deallocate();
    for (int i=0; i<rank; i++) {
      offsets  [i] = rhs.offsets  [i];
      lbounds  [i] = rhs.lbounds  [i];
      dimension[i] = rhs.dimension[i];
    }
    #ifdef ARRAY_DEBUG
      myname = rhs.myname;
    #endif
    myData   = rhs.myData;
    refCount = rhs.refCount;
    if (owned) { (*refCount)++; }

    return *this;
  }


  /*
  MOVE CONSTRUCTORS
  This straight up steals the pointers form the rhs and sets them to null.
  */
  Array(Array &&rhs) {
    nullify();
    owned    = rhs.owned;
    for (int i=0; i<rank; i++) {
      offsets  [i] = rhs.offsets  [i];
      lbounds  [i] = rhs.lbounds  [i];
      dimension[i] = rhs.dimension[i];
    }
    #ifdef ARRAY_DEBUG
      myname = rhs.myname;
    #endif
    myData   = rhs.myData;
    refCount = rhs.refCount;

    rhs.myData   = nullptr;
    rhs.refCount = nullptr;
  }


  Array& operator=(Array &&rhs) {
    if (this == &rhs) {
      return *this;
    }
    owned    = rhs.owned;
    deallocate();
    for (int i=0; i<rank; i++) {
      offsets  [i] = rhs.offsets  [i];
      lbounds  [i] = rhs.lbounds  [i];
      dimension[i] = rhs.dimension[i];
    }
    #ifdef ARRAY_DEBUG
      myname = rhs.myname;
    #endif
    myData   = rhs.myData;
    refCount = rhs.refCount;

    rhs.myData   = nullptr;
    rhs.refCount = nullptr;

    return *this;
  }


  /*
  DESTRUCTOR
  Decrement the refCounter, and if it's zero, deallocate and nullify.  
  */
  ~Array() {
    deallocate();
  }


  /* SETUP FUNCTIONS
  Initialize the array with the given dimensions
  */
  inline void setup(char const * label, int const d1) {
    static_assert( rank == 1 , "ERROR: Calling invalid function on rank 1 Array" );
    int tmp[1];
    tmp[0] = d1;
    setup_arr(label, tmp);
  }
  inline void setup(char const * label, int const d1, int const d2) {
    static_assert( rank == 2 , "ERROR: Calling invalid function on rank 2 Array" );
    int tmp[2];
    tmp[0] = d1;
    tmp[1] = d2;
    setup_arr(label, tmp);
  }
  inline void setup(char const * label, int const d1, int const d2, int const d3) {
    static_assert( rank == 3 , "ERROR: Calling invalid function on rank 3 Array" );
    int tmp[3];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    setup_arr(label, tmp);
  }
  inline void setup(char const * label, int const d1, int const d2, int const d3, int const d4) {
    static_assert( rank == 4 , "ERROR: Calling invalid function on rank 4 Array" );
    int tmp[4];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    tmp[3] = d4;
    setup_arr(label, tmp);
  }
  inline void setup(char const * label, int const d1, int const d2, int const d3, int const d4, int const d5) {
    static_assert( rank == 5 , "ERROR: Calling invalid function on rank 5 Array" );
    int tmp[5];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    tmp[3] = d4;
    tmp[4] = d5;
    setup_arr(label, tmp);
  }
  inline void setup(char const * label, int const d1, int const d2, int const d3, int const d4, int const d5, int const d6) {
    static_assert( rank == 6 , "ERROR: Calling invalid function on rank 6 Array" );
    int tmp[6];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    tmp[3] = d4;
    tmp[4] = d5;
    tmp[5] = d6;
    setup_arr(label, tmp);
  }
  inline void setup(char const * label, int const d1, int const d2, int const d3, int const d4, int const d5, int const d6, int const d7) {
    static_assert( rank == 7 , "ERROR: Calling invalid function on rank 7 Array" );
    int tmp[7];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    tmp[3] = d4;
    tmp[4] = d5;
    tmp[5] = d6;
    tmp[6] = d7;
    setup_arr(label, tmp);
  }
  inline void setup(char const * label, int const d1, int const d2, int const d3, int const d4, int const d5, int const d6, int const d7, int const d8) {
    static_assert( rank == 8 , "ERROR: Calling invalid function on rank 8 Array" );
    int tmp[8];
    tmp[0] = d1;
    tmp[1] = d2;
    tmp[2] = d3;
    tmp[3] = d4;
    tmp[4] = d5;
    tmp[5] = d6;
    tmp[6] = d7;
    tmp[7] = d8;
    setup_arr(label, tmp);
  }
  inline void setup_arr(char const * label, int const dimension[]) {
    #ifdef ARRAY_DEBUG
      myname = std::string(label);
    #endif

    deallocate();

    // Setup this Array with the given number of dimensions and dimension sizes
    for (int i=0; i<rank; i++) {
      this->dimension[i] = dimension[i];
    }
    offsets[0] = 1;
    for (int i=1; i<rank; i++) {
      offsets[i] = offsets[i-1] * dimension[i-1];
    }
    allocate();
  }


  inline void allocate() {
    if (owned) {
      refCount = new int;
      *refCount = 1;
      if (myMem == memDevice) {
        myData = (T *) yaklAllocDevice( totElems()*sizeof(T) );
      } else {
        myData = (T *) yaklAllocHost  ( totElems()*sizeof(T) );
      }
    }
  }


  inline void deallocate() {
    if (owned) {
      if (refCount != nullptr) {
        (*refCount)--;

        if (*refCount == 0) {
          delete refCount;
          refCount = nullptr;
          if (myMem == memDevice) {
            yaklFreeDevice(myData);
          } else {
            yaklFreeHost  (myData);
          }
          myData = nullptr;
        }

      }
    }
  }


  /* ARRAY INDEXERS (FORTRAN index ordering)
  Return the element at the given index (either read-only or read-write)
  */
  YAKL_INLINE T &operator()(int const i0) const {
    static_assert( rank == 1 , "ERROR: Calling invalid function on rank 1 Array" );
    #ifdef ARRAY_DEBUG
      this->check_index(0,i0,lbounds[0],lbounds[0]+dimension[0]-1,__FILE__,__LINE__);
    #endif
    int ind = i0-lbounds[0];
    return myData[ind];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1) const {
    static_assert( rank == 2 , "ERROR: Calling invalid function on rank 2 Array" );
    #ifdef ARRAY_DEBUG
      this->check_index(0,i0,lbounds[0],lbounds[0]+dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,lbounds[1],lbounds[1]+dimension[1]-1,__FILE__,__LINE__);
    #endif
    int ind = (i1-lbounds[1])*offsets[1] +
              (i0-lbounds[0]);
    return myData[ind];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1, int const i2) const {
    static_assert( rank == 3 , "ERROR: Calling invalid function on rank 3 Array" );
    #ifdef ARRAY_DEBUG
      this->check_index(0,i0,lbounds[0],lbounds[0]+dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,lbounds[1],lbounds[1]+dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,lbounds[2],lbounds[2]+dimension[2]-1,__FILE__,__LINE__);
    #endif
    int ind = (i2-lbounds[2])*offsets[2] +
              (i1-lbounds[1])*offsets[1] +
              (i0-lbounds[0]);
    return myData[ind];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1, int const i2, int const i3) const {
    static_assert( rank == 4 , "ERROR: Calling invalid function on rank 4 Array" );
    #ifdef ARRAY_DEBUG
      this->check_index(0,i0,lbounds[0],lbounds[0]+dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,lbounds[1],lbounds[1]+dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,lbounds[2],lbounds[2]+dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,lbounds[3],lbounds[3]+dimension[3]-1,__FILE__,__LINE__);
    #endif
    int ind = (i3-lbounds[3])*offsets[3] +
              (i2-lbounds[2])*offsets[2] +
              (i1-lbounds[1])*offsets[1] +
              (i0-lbounds[0]);
    return myData[ind];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1, int const i2, int const i3, int const i4) const {
    static_assert( rank == 5 , "ERROR: Calling invalid function on rank 5 Array" );
    #ifdef ARRAY_DEBUG
      this->check_index(0,i0,lbounds[0],lbounds[0]+dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,lbounds[1],lbounds[1]+dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,lbounds[2],lbounds[2]+dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,lbounds[3],lbounds[3]+dimension[3]-1,__FILE__,__LINE__);
      this->check_index(4,i4,lbounds[4],lbounds[4]+dimension[4]-1,__FILE__,__LINE__);
    #endif
    int ind = (i4-lbounds[4])*offsets[4] +
              (i3-lbounds[3])*offsets[3] +
              (i2-lbounds[2])*offsets[2] +
              (i1-lbounds[1])*offsets[1] +
              (i0-lbounds[0]);
    return myData[ind];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1, int const i2, int const i3, int const i4, int const i5) const {
    static_assert( rank == 6 , "ERROR: Calling invalid function on rank 6 Array" );
    #ifdef ARRAY_DEBUG
      this->check_index(0,i0,lbounds[0],lbounds[0]+dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,lbounds[1],lbounds[1]+dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,lbounds[2],lbounds[2]+dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,lbounds[3],lbounds[3]+dimension[3]-1,__FILE__,__LINE__);
      this->check_index(4,i4,lbounds[4],lbounds[4]+dimension[4]-1,__FILE__,__LINE__);
      this->check_index(5,i5,lbounds[5],lbounds[5]+dimension[5]-1,__FILE__,__LINE__);
    #endif
    int ind = (i5-lbounds[5])*offsets[5] +
              (i4-lbounds[4])*offsets[4] +
              (i3-lbounds[3])*offsets[3] +
              (i2-lbounds[2])*offsets[2] +
              (i1-lbounds[1])*offsets[1] +
              (i0-lbounds[0]);
    return myData[ind];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1, int const i2, int const i3, int const i4, int const i5, int const i6) const {
    static_assert( rank == 7 , "ERROR: Calling invalid function on rank 7 Array" );
    #ifdef ARRAY_DEBUG
      this->check_index(0,i0,lbounds[0],lbounds[0]+dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,lbounds[1],lbounds[1]+dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,lbounds[2],lbounds[2]+dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,lbounds[3],lbounds[3]+dimension[3]-1,__FILE__,__LINE__);
      this->check_index(4,i4,lbounds[4],lbounds[4]+dimension[4]-1,__FILE__,__LINE__);
      this->check_index(5,i5,lbounds[5],lbounds[5]+dimension[5]-1,__FILE__,__LINE__);
      this->check_index(6,i6,lbounds[6],lbounds[6]+dimension[6]-1,__FILE__,__LINE__);
    #endif
    int ind = (i6-lbounds[6])*offsets[6] +
              (i5-lbounds[5])*offsets[5] +
              (i4-lbounds[4])*offsets[4] +
              (i3-lbounds[3])*offsets[3] +
              (i2-lbounds[2])*offsets[2] +
              (i1-lbounds[1])*offsets[1] +
              (i0-lbounds[0]);
    return myData[ind];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1, int const i2, int const i3, int const i4, int const i5, int const i6, int const i7) const {
    static_assert( rank == 8 , "ERROR: Calling invalid function on rank 8 Array" );
    #ifdef ARRAY_DEBUG
      this->check_index(0,i0,lbounds[0],lbounds[0]+dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,lbounds[1],lbounds[1]+dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,lbounds[2],lbounds[2]+dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,lbounds[3],lbounds[3]+dimension[3]-1,__FILE__,__LINE__);
      this->check_index(4,i4,lbounds[4],lbounds[4]+dimension[4]-1,__FILE__,__LINE__);
      this->check_index(5,i5,lbounds[5],lbounds[5]+dimension[5]-1,__FILE__,__LINE__);
      this->check_index(6,i6,lbounds[6],lbounds[6]+dimension[6]-1,__FILE__,__LINE__);
      this->check_index(7,i7,lbounds[7],lbounds[7]+dimension[7]-1,__FILE__,__LINE__);
    #endif
    int ind = (i7-lbounds[7])*offsets[7] +
              (i6-lbounds[6])*offsets[6] +
              (i5-lbounds[5])*offsets[5] +
              (i4-lbounds[4])*offsets[4] +
              (i3-lbounds[3])*offsets[3] +
              (i2-lbounds[2])*offsets[2] +
              (i1-lbounds[1])*offsets[1] +
              (i0-lbounds[0]);
    return myData[ind];
  }


  inline void check_index(int const dim, long const ind, long const lb, long const ub, char const *file, int const line) const {
    #ifdef ARRAY_DEBUG
    if (ind < lb || ind > ub) {
      std::stringstream ss;
      ss << "For Array labeled: " << myname << "\n";
      ss << "Index " << dim << " of " << this->rank << " out of bounds\n";
      ss << "File, Line: " << file << ", " << line << "\n";
      ss << "Index: " << ind << ". Bounds: (" << lb << "," << ub << ")\n";
      throw std::out_of_range(ss.str());
    }
    #endif
  }


  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( Dims const &dims ) const {
    Array<T,N,myMem,styleFortran> ret;
    ret.owned = false;
    for (int i=0; i<N; i++) {
      ret.dimension[i] = dimension[i];
      ret.offsets  [i] = offsets  [i];
      ret.lbounds  [i] = lbounds  [i];
    }
    size_t retOff = 0;
    for (int i=N; i<rank; i++) {
      retOff += (dims.data[i]-lbounds[i])*offsets[i];
    }
    ret.myData = &(this->myData[retOff]);
    return ret;
  }


  inline Array<T,rank,memHost,styleFortran> createHostCopy() {
    Array<T,rank,memHost,styleFortran> ret;
    #ifdef ARRAY_DEBUG
      ret.setup_arr( myname.c_str() , dimension );
    #else
      ret.setup_arr( ""             , dimension );
    #endif
    if (myMem == memHost) {
      for (size_t i=0; i<totElems(); i++) {
        ret.myData[i] = myData[i];
      }
    } else {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),cudaMemcpyDeviceToHost,0);
        cudaDeviceSynchronize();
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),hipMemcpyDeviceToHost,0);
        hipDeviceSynchronize();
      #endif
    }
    return ret;
  }


  inline Array<T,rank,memDevice,styleFortran> createDeviceCopy() {
    Array<T,rank,memDevice,styleFortran> ret;
    #ifdef ARRAY_DEBUG
      ret.setup_arr( myname.c_str() , dimension );
    #else
      ret.setup_arr( ""             , dimension );
    #endif
    if (myMem == memHost) {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),cudaMemcpyHostToDevice,0);
        cudaDeviceSynchronize();
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),hipMemcpyHostToDevice,0);
        hipDeviceSynchronize();
      #endif
    } else {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),cudaMemcpyDeviceToDevice,0);
        cudaDeviceSynchronize();
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),hipMemcpyDeviceToDevice,0);
        hipDeviceSynchronize();
      #endif
    }
    return ret;
  }


  inline void deep_copy_to(Array<T,rank,memHost,styleFortran> lhs) {
    if (myMem == memHost) {
      for (size_t i=0; i<totElems(); i++) { lhs.myData[i] = myData[i]; }
    } else {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),cudaMemcpyDeviceToHost,0);
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),hipMemcpyDeviceToHost,0);
      #else
        for (size_t i=0; i<totElems(); i++) { lhs.myData[i] = myData[i]; }
      #endif
    }
  }


  inline void deep_copy_to(Array<T,rank,memDevice,styleFortran> lhs) {
    if (myMem == memHost) {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),cudaMemcpyHostToDevice,0);
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),hipMemcpyHostToDevice,0);
      #else
        for (size_t i=0; i<totElems(); i++) { lhs.myData[i] = myData[i]; }
      #endif
    } else {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),cudaMemcpyDeviceToDevice,0);
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),hipMemcpyDeviceToDevice,0);
      #else
        for (size_t i=0; i<totElems(); i++) { lhs.myData[i] = myData[i]; }
      #endif
    }
  }


  inline T sum() {
    T sum = 0;
    for (int i=0; i<totElems(); i++) { sum += myData[i]; }
    return sum;
  }


  void setRandom() {
    Random rand;
    rand.fillArray(this->data(),this->totElems());
  }
  void setRandom(Random &rand) {
    rand.fillArray(this->data(),this->totElems());
  }


  T relNorm(Array<T,rank,myMem,styleFortran> &other) {
    double numer = 0;
    double denom = 0;
    for (int i=0; i<this->totElems(); i++) {
      numer += abs(this->myData[i] - other.myData[i]);
      denom += abs(this->myData[i]);
    }
    if (denom > 0) {
      numer /= denom;
    }
    return numer;
  }


  T absNorm(Array<T,rank,myMem,styleFortran> &other) {
    T numer = 0;
    for (int i=0; i<this->totElems(); i++) {
      numer += abs(this->myData[i] - other.myData[i]);
    }
    return numer;
  }


  T maxAbs(Array<T,rank,myMem,styleFortran> &other) {
    T numer = abs(this->myData[0] - other.myData[0]);
    for (int i=1; i<this->totElems(); i++) {
      numer = max( numer , abs(this->myData[i] - other.myData[i]) );
    }
    return numer;
  }


  /* ACCESSORS */
  YAKL_INLINE int get_rank() const {
    return rank;
  }
  YAKL_INLINE size_t totElems() const {
    size_t totElems = dimension[0];
    for (int i=1; i<rank; i++) {
      totElems *= dimension[i];
    }
    return totElems;
  }
  YAKL_INLINE size_t get_totElems() const {
    return totElems();
  }
  YAKL_INLINE int const *get_dimensions() const {
    return dimension;
  }
  YAKL_INLINE T *data() const {
    return myData;
  }
  YAKL_INLINE T *get_data() const {
    return myData;
  }
  YAKL_INLINE int extent( int const dim ) const {
    return dimension[dim];
  }
  YAKL_INLINE int extent_int( int const dim ) const {
    return (int) dimension[dim];
  }

  YAKL_INLINE int span_is_contiguous() const {
    return 1;
  }
  YAKL_INLINE int use_count() const {
    if (owned) {
      return *refCount;
    } else {
      return -1;
    }
  }
  #ifdef ARRAY_DEBUG
    const char* label() const {
      return myname.c_str();
    }
  #endif


  /* INFORM */
  inline void print_rank() const {
    #ifdef ARRAY_DEBUG
      std::cout << "For Array labeled: " << myname << "\n";
    #endif
    std::cout << "Number of Dimensions: " << rank << "\n";
  }
  inline void print_totElems() const {
    #ifdef ARRAY_DEBUG
      std::cout << "For Array labeled: " << myname << "\n";
    #endif
    std::cout << "Total Number of Elements: " << totElems() << "\n";
  }
  inline void print_dimensions() const {
    #ifdef ARRAY_DEBUG
      std::cout << "For Array labeled: " << myname << "\n";
    #endif
    std::cout << "Dimension Sizes: ";
    for (int i=0; i<rank; i++) {
      std::cout << dimension[i] << ", ";
    }
    std::cout << "\n";
  }
  inline void print_data() const {
    #ifdef ARRAY_DEBUG
      std::cout << "For Array labeled: " << myname << "\n";
    #endif
    if (rank == 1) {
      for (int i=0; i<dimension[0]; i++) {
        std::cout << std::setw(12) << (*this)(i) << "\n";
      }
    } else if (rank == 2) {
      for (int j=0; j<dimension[0]; j++) {
        for (int i=0; i<dimension[1]; i++) {
          std::cout << std::setw(12) << (*this)(i,j) << " ";
        }
        std::cout << "\n";
      }
    } else if (rank == 0) {
      std::cout << "Empty Array\n\n";
    } else {
      for (size_t i=0; i<totElems(); i++) {
        std::cout << std::setw(12) << myData[i] << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }


  /* OPERATOR<<
  Print the array. If it's 2-D, print a pretty looking matrix */
  inline friend std::ostream &operator<<(std::ostream& os, Array const &v) {
    #ifdef ARRAY_DEBUG
      os << "For Array labeled: " << v.myname << "\n";
    #endif
    os << "Number of Dimensions: " << v.rank << "\n";
    os << "Total Number of Elements: " << v.totElems() << "\n";
    os << "Dimension Sizes: ";
    for (int i=0; i<v.rank; i++) {
      os << v.dimension[i] << ", ";
    }
    os << "\n";
    if (v.rank == 1) {
      for (int i=0; i<v.dimension[0]; i++) {
        os << std::setw(12) << v(i) << "\n";
      }
    } else if (v.rank == 2) {
      for (int j=0; j<v.dimension[1]; j++) {
        for (int i=0; i<v.dimension[0]; i++) {
          os << std::setw(12) << v(i,j) << " ";
        }
        os << "\n";
      }
    } else if (v.rank == 0) {
      os << "Empty Array\n\n";
    } else {
      for (size_t i=0; i<v.totElems(); i++) {
        os << v.myData[i] << " ";
      }
      os << "\n";
    }
    os << "\n";
    return os;
  }


};


