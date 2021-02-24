
#pragma once

template <class T, int rank, int myMem> class Array<T,rank,myMem,styleC> {
public:

  index_t offsets  [rank];  // Precomputed dimension offsets for efficient data access into a 1-D pointer
  index_t dimension[rank];  // Sizes of the 8 possible dimensions
  T       * myData;         // Pointer to the flattened internal data
  int     * refCount;       // Pointer shared by multiple copies of this Array to keep track of allcation / free
  bool    owned;            // Whether is is owned (owned = allocated,ref_counted,deallocated) or not

  // Start off all constructors making sure the pointers are null
  YAKL_INLINE void nullify() {
    owned    = true;
    myData   = nullptr;
    refCount = nullptr;
  }

  /* CONSTRUCTORS
  Always nullify before beginning so that myData == nullptr upon init.
  */
  YAKL_INLINE Array() {
    nullify();
  }
  YAKL_INLINE Array(char const * label) {
    nullify();
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Owned constructors
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Array(char const * label, index_t const d1) {
    nullify();
    deallocate();
    dimension[0] = d1;
    compute_offsets();
    allocate(label);
  }

  template <class INT, typename std::enable_if< std::is_integral<INT>::value , int >::type = 0>
  Array(char const * label, std::vector<INT> const dims) {
    nullify();
    deallocate();
    for (int i=0; i < rank; i++) {
      dimension[i] = dims[i];
    }
    compute_offsets();
    allocate(label);
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Non-owned constructors
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Array(char const * label, T * data, index_t const d1) {
    nullify();
    owned = false;
    dimension[0] = d1;
    compute_offsets();
    myData = data;
  }

  template <class INT, typename std::enable_if< std::is_integral<INT>::value , int >::type = 0>
  Array(char const * label, T * data, std::vector<INT> const dims) {
    nullify();
    owned = false;
    for (int i=0; i < rank; i++) {
      dimension[i] = dims[i];
    }
    compute_offsets();
    myData = data;
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
  YAKL_INLINE T &operator()(index_t const i0) const {
    index_t ind = i0;
    return myData[ind];
  }

  inline void check_index(int const dim, long const ind, long const lb, long const ub, char const *file, int const line) const {
    if (ind < lb || ind > ub) {
      std::cout << "Index " << dim+1 << " of " << rank << " out of bounds\n";
      std::cout << "File, Line: " << file << ", " << line << "\n";
      std::cout << "Index: " << ind << ". Bounds: (" << lb << "," << ub << ")\n";
      throw "";
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
  YAKL_INLINE index_t totElems() const {
    return get_totElems();
  }
  YAKL_INLINE T *data() const {
    return myData;
  }

  inline void compute_offsets() {
    offsets[rank-1] = 1;
    for (int i=rank-2; i>=0; i--) {
      offsets[i] = offsets[i+1] * dimension[i+1];
    }
  }

  inline void allocate(char const * label = "") {
    if (owned) {
      refCount = new int;
      *refCount = 1;
      //myData = (T *) yaklAllocDevice( totElems()*sizeof(T) , label );
      //myData = sycl::malloc_device(bytes,sycl_default_stream);
      myData = (T *) sycl::malloc_device(totElems()*sizeof(T),sycl_default_stream);
    }
  }


  YAKL_INLINE void deallocate() {
    if (owned) {
      if (refCount != nullptr) {
        (*refCount)--;

        if (*refCount == 0) {
          delete refCount;
          refCount = nullptr;
          if (totElems() > 0) {
            //yaklFreeDevice(myData,"");
            sycl::free(myData, sycl_default_stream);
            myData = nullptr;
          }
        }
      }
    }
  }

};

