
#pragma once



// Dynamic (runtime) Array Bounds
class Bnd {
public:
  int l, u;
  Bnd(                  ) { l = 1   ; u = 1   ; }
  Bnd(          int u_in) { l = 1   ; u = u_in; }
  Bnd(int l_in, int u_in) { l = l_in; u = u_in; }
};



template <class T, int rank, int myMem> class Array<T,rank,myMem,styleFortran> {
public:

  index_t offsets  [rank];  // Precomputed dimension offsets for efficient data access into a 1-D pointer
  int     lbounds  [rank];  // Lower bounds for each dimension
  index_t dimension[rank];  // Sizes of dimensions
  T       * myData;         // Pointer to the flattened internal data
  int     * refCount;       // Pointer shared by multiple copies of this Array to keep track of allcation / free
  bool    owned;            // Whether is is owned (owned = allocated,ref_counted,deallocated) or not
  #ifdef YAKL_DEBUG
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
    #ifdef YAKL_DEBUG
      myname = std::string(label);
    #endif
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Owned constructors
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Array(char const * label, Bnd const &b1) {
    #ifdef YAKL_DEBUG
      if ( rank != 1 ) { yakl_throw("ERROR: Calling invalid constructor on rank 1 Array"); }
    #endif
    nullify();
    setup(label,b1);
  }
  Array(char const * label, Bnd const &b1, Bnd const &b2) {
    #ifdef YAKL_DEBUG
      if ( rank != 2 ) { yakl_throw("ERROR: Calling invalid constructor on rank 2 Array"); }
    #endif
    nullify();
    setup(label,b1,b2);
  }
  Array(char const * label, Bnd const &b1, Bnd const &b2, Bnd const &b3) {
    #ifdef YAKL_DEBUG
      if ( rank != 3 ) { yakl_throw("ERROR: Calling invalid constructor on rank 3 Array"); }
    #endif
    nullify();
    setup(label,b1,b2,b3);
  }
  Array(char const * label, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4) {
    #ifdef YAKL_DEBUG
      if ( rank != 4 ) { yakl_throw("ERROR: Calling invalid constructor on rank 4 Array"); }
    #endif
    nullify();
    setup(label,b1,b2,b3,b4);
  }
  Array(char const * label, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5) {
    #ifdef YAKL_DEBUG
      if ( rank != 5 ) { yakl_throw("ERROR: Calling invalid constructor on rank 5 Array"); }
    #endif
    nullify();
    setup(label,b1,b2,b3,b4,b5);
  }
  Array(char const * label, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5, Bnd const &b6) {
    #ifdef YAKL_DEBUG
      if ( rank != 6 ) { yakl_throw("ERROR: Calling invalid constructor on rank 6 Array"); }
    #endif
    nullify();
    setup(label,b1,b2,b3,b4,b5,b6);
  }
  Array(char const * label, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5, Bnd const &b6, Bnd const &b7) {
    #ifdef YAKL_DEBUG
      if ( rank != 7 ) { yakl_throw("ERROR: Calling invalid constructor on rank 7 Array"); }
    #endif
    nullify();
    setup(label,b1,b2,b3,b4,b5,b6,b7);
  }
  Array(char const * label, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5, Bnd const &b6, Bnd const &b7, Bnd const &b8) {
    #ifdef YAKL_DEBUG
      if ( rank != 8 ) { yakl_throw("ERROR: Calling invalid constructor on rank 8 Array"); }
    #endif
    nullify();
    setup(label,b1,b2,b3,b4,b5,b6,b7,b8);
  }
  Array(char const * label, std::vector<Bnd> const &bnds) {
    #ifdef YAKL_DEBUG
      if ( bnds.size() < rank ) { yakl_throw("ERROR: Number of array bounds specified is < rank"); }
      if ( rank < 1 || rank > 8) { yakl_throw("ERROR: Invalid number of ranks. Must be between 1 and 8"); }
    #endif
    nullify();
         if ( rank == 1 ) { setup(label,bnds[0]); }
    else if ( rank == 2 ) { setup(label,bnds[0],bnds[1]); } 
    else if ( rank == 3 ) { setup(label,bnds[0],bnds[1],bnds[2]); } 
    else if ( rank == 4 ) { setup(label,bnds[0],bnds[1],bnds[2],bnds[3]); } 
    else if ( rank == 5 ) { setup(label,bnds[0],bnds[1],bnds[2],bnds[3],bnds[4]); } 
    else if ( rank == 6 ) { setup(label,bnds[0],bnds[1],bnds[2],bnds[3],bnds[4],bnds[5]); } 
    else if ( rank == 7 ) { setup(label,bnds[0],bnds[1],bnds[2],bnds[3],bnds[4],bnds[5],bnds[6]); } 
    else if ( rank == 8 ) { setup(label,bnds[0],bnds[1],bnds[2],bnds[3],bnds[4],bnds[5],bnds[6],bnds[7]); } 
  }
  template <class INT, typename std::enable_if< std::is_integral<INT>::value , int >::type = 0>
  Array(char const * label, std::vector<INT> const &bnds) {
    #ifdef YAKL_DEBUG
      if ( bnds.size() < rank ) { yakl_throw("ERROR: Number of array bounds specified is < rank"); }
      if ( rank < 1 || rank > 8) { yakl_throw("ERROR: Invalid number of ranks. Must be between 1 and 8"); }
    #endif
    nullify();
         if ( rank == 1 ) { setup(label,bnds[0]); }
    else if ( rank == 2 ) { setup(label,bnds[0],bnds[1]); } 
    else if ( rank == 3 ) { setup(label,bnds[0],bnds[1],bnds[2]); } 
    else if ( rank == 4 ) { setup(label,bnds[0],bnds[1],bnds[2],bnds[3]); } 
    else if ( rank == 5 ) { setup(label,bnds[0],bnds[1],bnds[2],bnds[3],bnds[4]); } 
    else if ( rank == 6 ) { setup(label,bnds[0],bnds[1],bnds[2],bnds[3],bnds[4],bnds[5]); } 
    else if ( rank == 7 ) { setup(label,bnds[0],bnds[1],bnds[2],bnds[3],bnds[4],bnds[5],bnds[6]); } 
    else if ( rank == 8 ) { setup(label,bnds[0],bnds[1],bnds[2],bnds[3],bnds[4],bnds[5],bnds[6],bnds[7]); } 
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Non-owned constructors
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Array(char const * label, T * data, Bnd const &b1) {
    #ifdef YAKL_DEBUG
      if ( rank != 1 ) { yakl_throw("ERROR: Calling invalid constructor on rank 1 Array"); }
    #endif
    nullify();
    owned = false;
    setup(label,b1);
    myData = data;
  }
  Array(char const * label, T * data, Bnd const &b1, Bnd const &b2) {
    #ifdef YAKL_DEBUG
      if ( rank != 2 ) { yakl_throw("ERROR: Calling invalid constructor on rank 2 Array"); }
    #endif
    nullify();
    owned = false;
    setup(label,b1,b2);
    myData = data;
  }
  Array(char const * label, T * data, Bnd const &b1, Bnd const &b2, Bnd const &b3) {
    #ifdef YAKL_DEBUG
      if ( rank != 3 ) { yakl_throw("ERROR: Calling invalid constructor on rank 3 Array"); }
    #endif
    nullify();
    owned = false;
    setup(label,b1,b2,b3);
    myData = data;
  }
  Array(char const * label, T * data, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4) {
    #ifdef YAKL_DEBUG
      if ( rank != 4 ) { yakl_throw("ERROR: Calling invalid constructor on rank 4 Array"); }
    #endif
    nullify();
    owned = false;
    setup(label,b1,b2,b3,b4);
    myData = data;
  }
  Array(char const * label, T * data, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5) {
    #ifdef YAKL_DEBUG
      if ( rank != 5 ) { yakl_throw("ERROR: Calling invalid constructor on rank 5 Array"); }
    #endif
    nullify();
    owned = false;
    setup(label,b1,b2,b3,b4,b5);
    myData = data;
  }
  Array(char const * label, T * data, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5, Bnd const &b6) {
    #ifdef YAKL_DEBUG
      if ( rank != 6 ) { yakl_throw("ERROR: Calling invalid constructor on rank 6 Array"); }
    #endif
    nullify();
    owned = false;
    setup(label,b1,b2,b3,b4,b5,b6);
    myData = data;
  }
  Array(char const * label, T * data, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5, Bnd const &b6, Bnd const &b7) {
    #ifdef YAKL_DEBUG
      if ( rank != 7 ) { yakl_throw("ERROR: Calling invalid constructor on rank 7 Array"); }
    #endif
    nullify();
    owned = false;
    setup(label,b1,b2,b3,b4,b5,b6,b7);
    myData = data;
  }
  Array(char const * label, T * data, Bnd const &b1, Bnd const &b2, Bnd const &b3, Bnd const &b4, Bnd const &b5, Bnd const &b6, Bnd const &b7, Bnd const &b8) {
    #ifdef YAKL_DEBUG
      if ( rank != 8 ) { yakl_throw("ERROR: Calling invalid constructor on rank 8 Array"); }
    #endif
    nullify();
    owned = false;
    setup(label,b1,b2,b3,b4,b5,b6,b7,b8);
    myData = data;
  }
  Array(char const * label, T * data, std::vector<Bnd> const &bnds) {
    #ifdef YAKL_DEBUG
      if ( bnds.size() < rank ) { yakl_throw("ERROR: Number of array bounds specified is < rank"); }
      if ( rank < 1 || rank > 8) { yakl_throw("ERROR: Invalid number of ranks. Must be between 1 and 8"); }
    #endif
    nullify();
    owned = false;
         if ( rank == 1 ) { setup(label,bnds[0]); }
    else if ( rank == 2 ) { setup(label,bnds[0],bnds[1]); } 
    else if ( rank == 3 ) { setup(label,bnds[0],bnds[1],bnds[2]); } 
    else if ( rank == 4 ) { setup(label,bnds[0],bnds[1],bnds[2],bnds[3]); } 
    else if ( rank == 5 ) { setup(label,bnds[0],bnds[1],bnds[2],bnds[3],bnds[4]); } 
    else if ( rank == 6 ) { setup(label,bnds[0],bnds[1],bnds[2],bnds[3],bnds[4],bnds[5]); } 
    else if ( rank == 7 ) { setup(label,bnds[0],bnds[1],bnds[2],bnds[3],bnds[4],bnds[5],bnds[6]); } 
    else if ( rank == 8 ) { setup(label,bnds[0],bnds[1],bnds[2],bnds[3],bnds[4],bnds[5],bnds[6],bnds[7]); } 
    myData = data;
  }
  template <class INT, typename std::enable_if< std::is_integral<INT>::value , int >::type = 0>
  Array(char const * label, T * data, std::vector<INT> const &bnds) {
    #ifdef YAKL_DEBUG
      if ( bnds.size() < rank ) { yakl_throw("ERROR: Number of array bounds specified is < rank"); }
      if ( rank < 1 || rank > 8) { yakl_throw("ERROR: Invalid number of ranks. Must be between 1 and 8"); }
    #endif
    nullify();
    owned = false;
         if ( rank == 1 ) { setup(label,bnds[0]); }
    else if ( rank == 2 ) { setup(label,bnds[0],bnds[1]); } 
    else if ( rank == 3 ) { setup(label,bnds[0],bnds[1],bnds[2]); } 
    else if ( rank == 4 ) { setup(label,bnds[0],bnds[1],bnds[2],bnds[3]); } 
    else if ( rank == 5 ) { setup(label,bnds[0],bnds[1],bnds[2],bnds[3],bnds[4]); } 
    else if ( rank == 6 ) { setup(label,bnds[0],bnds[1],bnds[2],bnds[3],bnds[4],bnds[5]); } 
    else if ( rank == 7 ) { setup(label,bnds[0],bnds[1],bnds[2],bnds[3],bnds[4],bnds[5],bnds[6]); } 
    else if ( rank == 8 ) { setup(label,bnds[0],bnds[1],bnds[2],bnds[3],bnds[4],bnds[5],bnds[6],bnds[7]); } 
    myData = data;
  }


  /*
  COPY CONSTRUCTORS / FUNCTIONS
  This shares the pointers with another Array and increments the refCounter
  */
  Array(Array const &rhs) {
    // This is a constructor, so no need to deallocate
    nullify();
    owned = rhs.owned;
    for (int i=0; i<rank; i++) {
      offsets  [i] = rhs.offsets  [i];
      lbounds  [i] = rhs.lbounds  [i];
      dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      myname = rhs.myname;
    #endif
    myData   = rhs.myData;
    refCount = rhs.refCount;
    if (owned) { (*refCount)++; }
  }


  Array & operator=(Array const &rhs) {
    if (this == &rhs) { return *this; }
    owned = rhs.owned;
    deallocate();
    for (int i=0; i<rank; i++) {
      offsets  [i] = rhs.offsets  [i];
      lbounds  [i] = rhs.lbounds  [i];
      dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      myname = rhs.myname;
    #endif
    myData   = rhs.myData;
    refCount = rhs.refCount;
    if (owned) { (*refCount)++; }

    return *this;
  }


  /*
  MOVE CONSTRUCTORS
  This steals the pointers form the rhs instead of sharing and sets rhs pointers to nullptr.
  Therefore, no need to increment reference counter
  */
  Array(Array &&rhs) {
    // This is a constructor, so no need to deallocate
    nullify();
    owned = rhs.owned;
    for (int i=0; i<rank; i++) {
      offsets  [i] = rhs.offsets  [i];
      lbounds  [i] = rhs.lbounds  [i];
      dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
      myname = rhs.myname;
    #endif
    myData   = rhs.myData;
    refCount = rhs.refCount;

    rhs.myData   = nullptr;
    rhs.refCount = nullptr;
  }


  Array& operator=(Array &&rhs) {
    if (this == &rhs) { return *this; }
    owned = rhs.owned;
    deallocate();
    for (int i=0; i<rank; i++) {
      offsets  [i] = rhs.offsets  [i];
      lbounds  [i] = rhs.lbounds  [i];
      dimension[i] = rhs.dimension[i];
    }
    #ifdef YAKL_DEBUG
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
  YAKL_INLINE ~Array() {
    deallocate();
  }


  /* ARRAY INDEXERS (FORTRAN index ordering)
  Return the element at the given index (either read-only or read-write)
  */
  YAKL_INLINE T &operator()(int const i0) const {
    #ifdef YAKL_DEBUG
      if ( rank != 1 ) { yakl_throw("ERROR: Calling invalid function on rank 1 Array"); }
      this->check_index(0,i0,lbounds[0],lbounds[0]+dimension[0]-1,__FILE__,__LINE__);
    #endif
    index_t ind = i0-lbounds[0];
    return myData[ind];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1) const {
    #ifdef YAKL_DEBUG
      if ( rank != 2 ) { yakl_throw("ERROR: Calling invalid function on rank 2 Array"); }
      this->check_index(0,i0,lbounds[0],lbounds[0]+dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,lbounds[1],lbounds[1]+dimension[1]-1,__FILE__,__LINE__);
    #endif
    index_t ind = (i1-lbounds[1])*offsets[1] +
                  (i0-lbounds[0]);
    return myData[ind];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1, int const i2) const {
    #ifdef YAKL_DEBUG
      if ( rank != 3 ) { yakl_throw("ERROR: Calling invalid function on rank 3 Array"); }
      this->check_index(0,i0,lbounds[0],lbounds[0]+dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,lbounds[1],lbounds[1]+dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,lbounds[2],lbounds[2]+dimension[2]-1,__FILE__,__LINE__);
    #endif
    index_t ind = (i2-lbounds[2])*offsets[2] +
                  (i1-lbounds[1])*offsets[1] +
                  (i0-lbounds[0]);
    return myData[ind];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1, int const i2, int const i3) const {
    #ifdef YAKL_DEBUG
      if ( rank != 4 ) { yakl_throw("ERROR: Calling invalid function on rank 4 Array"); }
      this->check_index(0,i0,lbounds[0],lbounds[0]+dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,lbounds[1],lbounds[1]+dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,lbounds[2],lbounds[2]+dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,lbounds[3],lbounds[3]+dimension[3]-1,__FILE__,__LINE__);
    #endif
    index_t ind = (i3-lbounds[3])*offsets[3] +
                  (i2-lbounds[2])*offsets[2] +
                  (i1-lbounds[1])*offsets[1] +
                  (i0-lbounds[0]);
    return myData[ind];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1, int const i2, int const i3, int const i4) const {
    #ifdef YAKL_DEBUG
      if ( rank != 5 ) { yakl_throw("ERROR: Calling invalid function on rank 5 Array"); }
      this->check_index(0,i0,lbounds[0],lbounds[0]+dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,lbounds[1],lbounds[1]+dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,lbounds[2],lbounds[2]+dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,lbounds[3],lbounds[3]+dimension[3]-1,__FILE__,__LINE__);
      this->check_index(4,i4,lbounds[4],lbounds[4]+dimension[4]-1,__FILE__,__LINE__);
    #endif
    index_t ind = (i4-lbounds[4])*offsets[4] +
                  (i3-lbounds[3])*offsets[3] +
                  (i2-lbounds[2])*offsets[2] +
                  (i1-lbounds[1])*offsets[1] +
                  (i0-lbounds[0]);
    return myData[ind];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1, int const i2, int const i3, int const i4, int const i5) const {
    #ifdef YAKL_DEBUG
      if ( rank != 6 ) { yakl_throw("ERROR: Calling invalid function on rank 6 Array"); }
      this->check_index(0,i0,lbounds[0],lbounds[0]+dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,lbounds[1],lbounds[1]+dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,lbounds[2],lbounds[2]+dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,lbounds[3],lbounds[3]+dimension[3]-1,__FILE__,__LINE__);
      this->check_index(4,i4,lbounds[4],lbounds[4]+dimension[4]-1,__FILE__,__LINE__);
      this->check_index(5,i5,lbounds[5],lbounds[5]+dimension[5]-1,__FILE__,__LINE__);
    #endif
    index_t ind = (i5-lbounds[5])*offsets[5] +
                  (i4-lbounds[4])*offsets[4] +
                  (i3-lbounds[3])*offsets[3] +
                  (i2-lbounds[2])*offsets[2] +
                  (i1-lbounds[1])*offsets[1] +
                  (i0-lbounds[0]);
    return myData[ind];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1, int const i2, int const i3, int const i4, int const i5, int const i6) const {
    #ifdef YAKL_DEBUG
      if ( rank != 7 ) { yakl_throw("ERROR: Calling invalid function on rank 7 Array"); }
      this->check_index(0,i0,lbounds[0],lbounds[0]+dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,lbounds[1],lbounds[1]+dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,lbounds[2],lbounds[2]+dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,lbounds[3],lbounds[3]+dimension[3]-1,__FILE__,__LINE__);
      this->check_index(4,i4,lbounds[4],lbounds[4]+dimension[4]-1,__FILE__,__LINE__);
      this->check_index(5,i5,lbounds[5],lbounds[5]+dimension[5]-1,__FILE__,__LINE__);
      this->check_index(6,i6,lbounds[6],lbounds[6]+dimension[6]-1,__FILE__,__LINE__);
    #endif
    index_t ind = (i6-lbounds[6])*offsets[6] +
                  (i5-lbounds[5])*offsets[5] +
                  (i4-lbounds[4])*offsets[4] +
                  (i3-lbounds[3])*offsets[3] +
                  (i2-lbounds[2])*offsets[2] +
                  (i1-lbounds[1])*offsets[1] +
                  (i0-lbounds[0]);
    return myData[ind];
  }
  YAKL_INLINE T &operator()(int const i0, int const i1, int const i2, int const i3, int const i4, int const i5, int const i6, int const i7) const {
    #ifdef YAKL_DEBUG
      if ( rank != 8 ) { yakl_throw("ERROR: Calling invalid function on rank 8 Array"); }
      this->check_index(0,i0,lbounds[0],lbounds[0]+dimension[0]-1,__FILE__,__LINE__);
      this->check_index(1,i1,lbounds[1],lbounds[1]+dimension[1]-1,__FILE__,__LINE__);
      this->check_index(2,i2,lbounds[2],lbounds[2]+dimension[2]-1,__FILE__,__LINE__);
      this->check_index(3,i3,lbounds[3],lbounds[3]+dimension[3]-1,__FILE__,__LINE__);
      this->check_index(4,i4,lbounds[4],lbounds[4]+dimension[4]-1,__FILE__,__LINE__);
      this->check_index(5,i5,lbounds[5],lbounds[5]+dimension[5]-1,__FILE__,__LINE__);
      this->check_index(6,i6,lbounds[6],lbounds[6]+dimension[6]-1,__FILE__,__LINE__);
      this->check_index(7,i7,lbounds[7],lbounds[7]+dimension[7]-1,__FILE__,__LINE__);
    #endif
    index_t ind = (i7-lbounds[7])*offsets[7] +
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
    if (ind < lb || ind > ub) {
      #ifdef YAKL_DEBUG
        std::cout << "For Array labeled: " << myname << "\n";
      #endif
      std::cout << "Index " << dim+1 << " of " << rank << " out of bounds\n";
      std::cout << "File, Line: " << file << ", " << line << "\n";
      std::cout << "Index: " << ind << ". Bounds: (" << lb << "," << ub << ")\n";
      throw "";
    }
  }


  template <int N> YAKL_INLINE void slice( Dims const &dims , Array<T,N,myMem,styleFortran> &store ) const {
    #ifdef YAKL_DEBUG
      if (rank != dims.size()) {
        yakl_throw( "ERROR: rank must be equal to dims.size()" );
      }
    #endif
    store.owned = false;
    for (int i=0; i<N; i++) {
      store.dimension[i] = dimension[i];
      store.offsets  [i] = offsets  [i];
      store.lbounds  [i] = lbounds  [i];
    }
    index_t retOff = 0;
    for (int i=N; i<rank; i++) {
      retOff += (dims.data[i]-lbounds[i])*offsets[i];
    }
    store.myData = &(this->myData[retOff]);
  }
  template <int N> YAKL_INLINE void slice( int i0 , Array<T,N,myMem,styleFortran> &store ) const {
    slice( {i0} , store );
  }
  template <int N> YAKL_INLINE void slice( int i0, int i1 , Array<T,N,myMem,styleFortran> &store ) const {
    slice( {i0,i1} , store );
  }
  template <int N> YAKL_INLINE void slice( int i0, int i1, int i2, Array<T,N,myMem,styleFortran> &store ) const {
    slice( {i0,i1,i2} , store );
  }
  template <int N> YAKL_INLINE void slice( int i0, int i1, int i2, int i3, Array<T,N,myMem,styleFortran> &store ) const {
    slice( {i0,i1,i2,i3} , store );
  }
  template <int N> YAKL_INLINE void slice( int i0, int i1, int i2, int i3, int i4, Array<T,N,myMem,styleFortran> &store ) const {
    slice( {i0,i1,i2,i3,i4} , store );
  }
  template <int N> YAKL_INLINE void slice( int i0, int i1, int i2, int i3, int i4, int i5, Array<T,N,myMem,styleFortran> &store ) const {
    slice( {i0,i1,i2,i3,i4,i5} , store );
  }
  template <int N> YAKL_INLINE void slice( int i0, int i1, int i2, int i3, int i4, int i5, int i6, Array<T,N,myMem,styleFortran> &store ) const {
    slice( {i0,i1,i2,i3,i4,i5,i6} , store );
  }
  template <int N> YAKL_INLINE void slice( int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, Array<T,N,myMem,styleFortran> &store ) const {
    slice( {i0,i1,i2,i3,i4,i5,i6,i7} , store );
  }


  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( Dims const &dims ) const {
    Array<T,N,myMem,styleFortran> ret;
    slice( dims , ret );
    return ret;
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0 ) const {
    return slice<N>( {i0} );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0, int i1 ) const {
    return slice<N>( {i0,i1} );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0, int i1, int i2 ) const {
    return slice<N>( {i0,i1,i2} );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0, int i1, int i2, int i3 ) const {
    return slice<N>( {i0,i1,i2,i3} );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0, int i1, int i2, int i3, int i4 ) const {
    return slice<N>( {i0,i1,i2,i3,i4} );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0, int i1, int i2, int i3, int i4, int i5 ) const {
    return slice<N>( {i0,i1,i2,i3,i4,i5} );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0, int i1, int i2, int i3, int i4, int i5, int i6 ) const {
    return slice<N>( {i0,i1,i2,i3,i4,i5,i6} );
  }
  template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7 ) const {
    return slice<N>( {i0,i1,i2,i3,i4,i5,i6,i7} );
  }


  inline Array<T,rank,memHost,styleFortran> createHostCopy() const {
    Array<T,rank,memHost,styleFortran> ret;  // nullified + owned == true
    for (int i=0; i<rank; i++) {
      ret.offsets  [i] = offsets  [i];
      ret.lbounds  [i] = lbounds  [i];
      ret.dimension[i] = dimension[i];
    }
    #ifdef YAKL_DEBUG
      ret.myname = myname;
    #endif
    ret.allocate();
    if (myMem == memHost) {
      for (index_t i=0; i<totElems(); i++) { ret.myData[i] = myData[i]; }
    } else {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),cudaMemcpyDeviceToHost,0);
        check_last_error();
        cudaDeviceSynchronize();
        check_last_error();
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),hipMemcpyDeviceToHost,0);
        check_last_error();
        hipDeviceSynchronize();
        check_last_error();
      #else
        for (index_t i=0; i<totElems(); i++) { ret.myData[i] = myData[i]; }
      #endif
    }
    return ret;
  }


  inline Array<T,rank,memDevice,styleFortran> createDeviceCopy() const {
    Array<T,rank,memDevice,styleFortran> ret;  // nullified + owned == true
    for (int i=0; i<rank; i++) {
      ret.offsets  [i] = offsets  [i];
      ret.lbounds  [i] = lbounds  [i];
      ret.dimension[i] = dimension[i];
    }
    #ifdef YAKL_DEBUG
      ret.myname = myname;
    #endif
    ret.allocate();
    if (myMem == memHost) {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),cudaMemcpyHostToDevice,0);
        check_last_error();
        cudaDeviceSynchronize();
        check_last_error();
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),hipMemcpyHostToDevice,0);
        check_last_error();
        hipDeviceSynchronize();
        check_last_error();
      #else
        for (index_t i=0; i<totElems(); i++) { ret.myData[i] = myData[i]; }
      #endif
    } else {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),cudaMemcpyDeviceToDevice,0);
        check_last_error();
        cudaDeviceSynchronize();
        check_last_error();
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(ret.myData,myData,totElems()*sizeof(T),hipMemcpyDeviceToDevice,0);
        check_last_error();
        hipDeviceSynchronize();
        check_last_error();
      #else
        for (index_t i=0; i<totElems(); i++) { ret.myData[i] = myData[i]; }
      #endif
    }
    return ret;
  }


  inline void deep_copy_to(Array<T,rank,memHost,styleFortran> lhs) const {
    if (myMem == memHost) {
      for (index_t i=0; i<totElems(); i++) { lhs.myData[i] = myData[i]; }
    } else {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),cudaMemcpyDeviceToHost,0);
        check_last_error();
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),hipMemcpyDeviceToHost,0);
        check_last_error();
      #else
        for (index_t i=0; i<totElems(); i++) { lhs.myData[i] = myData[i]; }
      #endif
    }
  }


  inline void deep_copy_to(Array<T,rank,memDevice,styleFortran> lhs) const {
    if (myMem == memHost) {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),cudaMemcpyHostToDevice,0);
        check_last_error();
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),hipMemcpyHostToDevice,0);
        check_last_error();
      #else
        for (index_t i=0; i<totElems(); i++) { lhs.myData[i] = myData[i]; }
      #endif
    } else {
      #ifdef __USE_CUDA__
        cudaMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),cudaMemcpyDeviceToDevice,0);
        check_last_error();
      #elif defined(__USE_HIP__)
        hipMemcpyAsync(lhs.myData,myData,totElems()*sizeof(T),hipMemcpyDeviceToDevice,0);
        check_last_error();
      #else
        for (index_t i=0; i<totElems(); i++) { lhs.myData[i] = myData[i]; }
      #endif
    }
  }


  /* ACCESSORS */
  YAKL_INLINE int get_rank() const {
    return rank;
  }
  YAKL_INLINE index_t get_totElems() const {
    index_t tot = dimension[0];
    for (int i=1; i<rank; i++) { tot *= dimension[i]; }
    return tot;
  }
  YAKL_INLINE index_t get_elem_count() const {
    return get_totElems();
  }
  YAKL_INLINE index_t totElems() const {
    return get_totElems();
  }
  YAKL_INLINE FSArray<index_t,1,SB<rank>> get_dimensions() const {
    FSArray<index_t,1,SB<rank>> ret;
    for (int i=0; i<rank; i++) { ret(i+1) = dimension[i]; }
    return ret;
  }
  YAKL_INLINE FSArray<int,1,SB<rank>> get_lbounds() const {
    FSArray<int,1,SB<rank>> ret;
    for (int i=0; i<rank; i++) { ret(i+1) = lbounds[i]; }
    return ret;
  }
  YAKL_INLINE FSArray<int,1,SB<rank>> get_ubounds() const {
    FSArray<int,1,SB<rank>> ret;
    for (int i=0; i<rank; i++) { ret(i+1) = lbounds[i]+dimension[i]-1; }
    return ret;
  }
  YAKL_INLINE T *data() const {
    return myData;
  }
  YAKL_INLINE T *get_data() const {
    return myData;
  }
  YAKL_INLINE index_t extent( int const dim ) const {
    return dimension[dim];
  }
  YAKL_INLINE bool span_is_contiguous() const {
    return true;
  }
  YAKL_INLINE int use_count() const {
    if (owned) {
      return *refCount;
    } else {
      return -1;
    }
  }
  YAKL_INLINE bool initialized() const {
    return myData != nullptr;
  }
  const char* label() const {
    #ifdef YAKL_DEBUG
      return myname.c_str();
    #else
      return "";
    #endif
  }


  /* OPERATOR<<
  Print the array. If it's 2-D, print a pretty looking matrix */
  inline friend std::ostream &operator<<(std::ostream& os, Array const &v) {
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
    for (index_t i=0; i<v.totElems(); i++) {
      os << v.myData[i] << " ";
    }
    os << "\n";
    return os;
  }



  inline void setup(char const * label, Bnd const &b1, Bnd const &b2=-1, Bnd const &b3=-1, Bnd const &b4=-1, Bnd const &b5=-1, Bnd const &b6=-1, Bnd const &b7=-1, Bnd const &b8=-1) {
    #ifdef YAKL_DEBUG
      myname = std::string(label);
    #endif

    deallocate();

                     lbounds[0] = b1.l; dimension[0] = b1.u - b1.l + 1;  
    if (rank >= 2) { lbounds[1] = b2.l; dimension[1] = b2.u - b2.l + 1; }
    if (rank >= 3) { lbounds[2] = b3.l; dimension[2] = b3.u - b3.l + 1; }
    if (rank >= 4) { lbounds[3] = b4.l; dimension[3] = b4.u - b4.l + 1; }
    if (rank >= 5) { lbounds[4] = b5.l; dimension[4] = b5.u - b5.l + 1; }
    if (rank >= 6) { lbounds[5] = b6.l; dimension[5] = b6.u - b6.l + 1; }
    if (rank >= 7) { lbounds[6] = b7.l; dimension[6] = b7.u - b7.l + 1; }
    if (rank >= 8) { lbounds[7] = b8.l; dimension[7] = b8.u - b8.l + 1; }

    offsets[0] = 1;
    for (int i=1; i<rank; i++) {
      offsets[i] = offsets[i-1] * dimension[i-1];
    }
    allocate(label);
  }


  inline void allocate(char const * label = "") {
    if (owned) {
      static_assert( std::is_arithmetic<T>() || myMem == memHost , 
                     "ERROR: You cannot use non-arithmetic types inside owned Arrays on the device" );
      refCount = new int;
      *refCount = 1;
      if (myMem == memDevice) {
        myData = (T *) yaklAllocDevice( totElems()*sizeof(T) , label);
      } else {
        myData = new T[totElems()];
      }
    }
  }


  YAKL_INLINE void deallocate() {
    if (owned) {
      if (refCount != nullptr) {
        (*refCount)--;

        if (*refCount == 0) {
          delete refCount;
          refCount = nullptr;
          if (myMem == memDevice) {
            #ifdef YAKL_DEBUG
              yaklFreeDevice(myData,myname.c_str());
            #else
              yaklFreeDevice(myData,"");
            #endif
          } else {
            delete[] myData;
          }
          myData = nullptr;
        }

      }
    }
  }

};


