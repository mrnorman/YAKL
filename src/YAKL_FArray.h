
#pragma once
// Included by YAKL_Array.h

__YAKL_NAMESPACE_WRAPPER_BEGIN__
namespace yakl {

  /** @brief This implements the yakl:Array class with yakl::styleFortran behavior
    *
    * yakl::styleFortran behavior means all lower bounds default to
    * one but can be set to artibrary values, and index ordering is column-major, meaning the left-most index varies the fastest.
    * IMPORTANT: Please see the yakl::ArrayBase class because this class includes all of its functionality.
    * click for more information.
    * @param T       The type of the array. For yakl::memHost arrays, this can be generally any time. For yakl::memDevice
    *                arrays, this needs to be a type with no constructor, preferably an arithmetic type.
    * @param rank    The number of dimensions for this array object.
    * @param myMem   The memory space for this array object: either yakl::memHost or yakl::memDevice
    * @param myStyle yakl::styleFortran
    *
    * YAKL's multi-dimensional array objects can be considered in some ways to behave as pointers. For instance, if you
    * assign one object to another:
    * ```
    * yakl::Array<float,1,memHost,styleFortra> arr1("arr1",n);
    * yakl::Array<float,1,memHost,styleFortra> arr2 = arr1; // This is very fast; no allocation or copy of underlying data occurs
    * arr2 = 5; // This will change both arr1 and arr2 because they share the same data pointer
    * ```
    * This will share the same data pointer between `arr1` and `arr2`. Further, passing yakl::Array objects by value is fast
    * because it only copies the metadata over and shares the pointer to the underlying data. 
    * 
    * To copy data to another array object's data pointer, use `deep_copy_to()`, `createHostCopy()`, or `createDeviceCopy()`.
    * ```
    * yakl::Array<float,1,memHost,styleFortra> arr1("arr1",n);
    * yakl::Array<float,1,memHost,styleFortra> arr2 = arr1.createHostCopy(); // This is slow. It allocates and copies underlying data.
    * arr2 = 5; // This will *not* change arr1 because they do not share the same data pointer.
    * ```
    *
    * A note on yakl::Array objects with a `const` underyling type `T`: You cannot **allocate** an array object with `const`
    * underlying type, but you can **create** one without allocating. You may copy an array object with non-`const` underlying
    * type to an array object with `const` underling type, but not vice versa. After copying an array of non-`const` type to
    * an array of `const` type, if you then alter the data pointed to by the non-`const` array, results are undefined. The
    * typical behavior is below:
    * ```
    * // Passing non-const type array as a parameter to a const-type array (this is fine)
    * real myfunc( Array<float const,1,memHost,styleFortra> arrConst ) { ... }
    * // ...
    * Array<float,1,memHost,styleFortra> arr("arr",n);
    * myfunc(arr); // Totally fine. myfunc just casts the underlying type to const giving const correctness when you want it.
    * // ...
    * Array<float const,1,memHost,styleFortra> arrConst;
    * arrConst = arr; // Again, this is totally fine, BUT, do not change the data pointed to by arr any more
    * arr.deallocate(); // This is the best practice. Once you transfer the data to a const type, deallocate the non-const type object
    * ```
    */
  template <class T, int rank, int myMem>
  class Array<T,rank,myMem,styleFortran> : public ArrayBase<T,rank,myMem,styleFortran> {
  public:

    /** @brief This is the type `T` without `const` and `volatile` modifiers */
    typedef typename std::remove_cv<T>::type       type;
    /** @brief This is the type `T` exactly as it was defined upon array object creation. */
    typedef          T                             value_type;
    /** @brief This is the type `T` with `const` added to it
      * @details If the original type has `volatile`, then so will this type. */
    typedef typename std::add_const<type>::type    const_value_type;
    /** @brief This is the type `T` with `const` removed from it 
      * @details If the original type has `volatile`, then so will this type. */
    typedef typename std::remove_const<type>::type non_const_value_type;

    //* @private */
    int lbounds[rank];  // Lower bounds


    // Start off all constructors making sure the pointers are null
    /** @private */
    YAKL_INLINE void nullify() {
      this->myData   = nullptr;
      this->refCount = nullptr;
      for (int i=0; i < rank; i++) { this->lbounds[i] = 1; this->dimension[i] = 0; }
      YAKL_EXECUTE_ON_HOST_ONLY( this->myname="Uninitialized"; )
    }

    /* CONSTRUCTORS
    Always nullify before beginning so that myData == nullptr upon init.
    */
    /** @brief Create an empty, unallocated object */
    YAKL_INLINE Array() {
      nullify();
    }
    /** @brief Create an empty, unallocated object with a label.*/
    YAKL_INLINE explicit Array(char const * label) {
      nullify();
      this->myname = label;
    }


    /** @class doxhide_FArray_owned_constructors
      * @brief dummy
      *
      * This is one of the yakl::FArray owned constructors.
      * Create and allocate an owned and reference counted array object.
      * For Fortran-style array objects, the left-most index varies the fastest. Constructor must match
      * the rank template argument.
      *        
      * @param label  String label for this array.
      * @param b[0-7] Bounds of the respective dimension.
      * @param bnds   yakl::Bnds object containing the dimension bounds. Must match the array rank.
      *
      * Exmaple usage:
      * ```
      * Array<float,1,memHost,styleFortran> arr("arr",n1);     // lower bound == 1; upper bound == n1
      * Array<float,1,memHost,styleFortran> arr("arr",{0,n1}); // lower bound == 0; upper bound == n1
      * Array<float const,4,memDevice,styleFortran> arr("arr",n1,n2,n3,n4);
      * Array<int,2,memHost,styleFortran> arr("arr",{ {0,n1} , {-1,n2+1} });
      * ```
      */


    /** @brief 1-D owned constructor
      * \copydetails doxhide_FArray_owned_constructors */
    YAKL_INLINE Array( char const* label , Bnd b1 ) : Array(label,Bnds(b1)) {
      static_assert( rank == 1 , "ERROR: Calling constructor with 1 bound on non-rank-1 array" );
    }
    /** @brief 2-D owned constructor
      * \copydetails doxhide_FArray_owned_constructors */
    YAKL_INLINE Array( char const* label , Bnd b1 ,
                                           Bnd b2 ) : Array(label,Bnds(b1,b2)) {
      static_assert( rank == 2 , "ERROR: Calling constructor with 2 bound on non-rank-2 array" );
    }
    /** @brief 3-D owned constructor
      * \copydetails doxhide_FArray_owned_constructors */
    YAKL_INLINE Array( char const* label , Bnd b1 ,
                                           Bnd b2 ,
                                           Bnd b3 ) : Array(label,Bnds(b1,b2,b3)) {
      static_assert( rank == 3 , "ERROR: Calling constructor with 3 bound on non-rank-3 array" );
    }
    /** @brief 4-D owned constructor
      * \copydetails doxhide_FArray_owned_constructors */
    YAKL_INLINE Array( char const* label , Bnd b1 ,
                                           Bnd b2 ,
                                           Bnd b3 ,
                                           Bnd b4 ) : Array(label,Bnds(b1,b2,b3,b4)) {
      static_assert( rank == 4 , "ERROR: Calling constructor with 4 bound on non-rank-4 array" );
    }
    /** @brief 5-D owned constructor
      * \copydetails doxhide_FArray_owned_constructors */
    YAKL_INLINE Array( char const* label , Bnd b1 ,
                                           Bnd b2 ,
                                           Bnd b3 ,
                                           Bnd b4 ,
                                           Bnd b5 ) : Array(label,Bnds(b1,b2,b3,b4,b5)) {
      static_assert( rank == 5 , "ERROR: Calling constructor with 5 bound on non-rank-5 array" );
    }
    /** @brief 6-D owned constructor
      * \copydetails doxhide_FArray_owned_constructors */
    YAKL_INLINE Array( char const* label , Bnd b1 ,
                                           Bnd b2 ,
                                           Bnd b3 ,
                                           Bnd b4 ,
                                           Bnd b5 ,
                                           Bnd b6 ) : Array(label,Bnds(b1,b2,b3,b4,b5,b6)) {
      static_assert( rank == 6 , "ERROR: Calling constructor with 6 bound on non-rank-6 array" );
    }
    /** @brief 7-D owned constructor
      * \copydetails doxhide_FArray_owned_constructors */
    YAKL_INLINE Array( char const* label , Bnd b1 ,
                                           Bnd b2 ,
                                           Bnd b3 ,
                                           Bnd b4 ,
                                           Bnd b5 ,
                                           Bnd b6 ,
                                           Bnd b7 ) : Array(label,Bnds(b1,b2,b3,b4,b5,b6,b7)) {
      static_assert( rank == 7 , "ERROR: Calling constructor with 7 bound on non-rank-7 array" );
    }
    /** @brief 8-D owned constructor
      * \copydetails doxhide_FArray_owned_constructors */
    YAKL_INLINE Array( char const* label , Bnd b1 ,
                                           Bnd b2 ,
                                           Bnd b3 ,
                                           Bnd b4 ,
                                           Bnd b5 ,
                                           Bnd b6 ,
                                           Bnd b7 ,
                                           Bnd b8 ) : Array(label,Bnds(b1,b2,b3,b4,b5,b6,b7,b8)) {
      static_assert( rank == 8 , "ERROR: Calling constructor with 8 bound on non-rank-8 array" );
    }
    /** @brief Generic initializer-list or std::vector based owned constructor
      * \copydetails doxhide_FArray_owned_constructors */
    YAKL_INLINE Array(char const * label, Bnds bnds) {
      static_assert( rank >= 1 && rank <= 8 , "ERROR: Creating Array with a rank < 1 or > 8" );
      nullify();
      #ifdef YAKL_DEBUG
        if ( bnds.size() < rank ) { yakl_throw("ERROR: Number of array bounds specified is < rank"); }
      #endif
      YAKL_EXECUTE_ON_HOST_ONLY( this->deallocate(); )
      this->myname = label;
      for (int i=0; i < rank; i++) { this->lbounds[i] = bnds[i].l; this->dimension[i] = bnds[i].u - bnds[i].l + 1; }
      YAKL_EXECUTE_ON_HOST_ONLY( this->allocate(); )
    }


    /** @class doxhide_FArray_non_owned_constructors
      * @brief dummy
      *
      * This is one of the yakl::FArray non-owned constructors.
      * Create a non-owned and non-reference-counted array object that wraps the
      * provided data pointer. 
      * For Fortran-style array objects, the left-most index varies the fastest. Constructor must match
      * the rank template argument.
      * When creating a non-owned array object using this form of constructor, it is up to the user to ensure
      * that the underlying data pointer remains allocationg while it is used by this array object.
      * Since this performs no allocations, this constructor may be called on the device, and it has very
      * little runtime cost associated with it.
      *        
      * @param label  String label for this array.
      * @param data   Pointer to the allocated data being wrapped by this non-owned array object
      * @param b[0-7] Bounds of the respective dimension.
      * @param bnds   yakl::Bnds object containing the dimensions bounds. Must match the array rank.
      *
      * Exmaple usage:
      * ```
      * Array<float,2,memDevice,styleFortran> arr_owned("arr_owned",{0:n1},n2); // Owned Constructor
      * Array<float,2,memDevice,styleFortran> arr1("arr1",arr_owned.data(),n1+1,n2);
      * Array<float,2,memDevice,styleFortran> arr2("arr2",arr_owned.data(),{n1,{0:n2-1}});
      * ```
      */
    /** @brief 1-D non-owned constructor
      * \copydetails doxhide_FArray_non_owned_constructors */
    YAKL_INLINE Array( char const *label , T *data, Bnd b1 ) : Array(label,data,Bnds(b1)) {
      static_assert( rank == 1 , "ERROR: Calling constructor with 1 bound on non-rank-1 array" );
    }
    /** @brief 2-D non-owned constructor
      * \copydetails doxhide_FArray_non_owned_constructors */
    YAKL_INLINE Array( char const *label , T *data, Bnd b1 ,
                                                    Bnd b2 ) : Array(label,data,Bnds(b1,b2)) {
      static_assert( rank == 2 , "ERROR: Calling constructor with 2 bound on non-rank-2 array" );
    }
    /** @brief 3-D non-owned constructor
      * \copydetails doxhide_FArray_non_owned_constructors */
    YAKL_INLINE Array( char const *label , T *data, Bnd b1 ,
                                                    Bnd b2 ,
                                                    Bnd b3 ) : Array(label,data,Bnds(b1,b2,b3)) {
      static_assert( rank == 3 , "ERROR: Calling constructor with 3 bound on non-rank-3 array" );
    }
    /** @brief 4-D non-owned constructor
      * \copydetails doxhide_FArray_non_owned_constructors */
    YAKL_INLINE Array( char const *label , T *data, Bnd b1 ,
                                                    Bnd b2 ,
                                                    Bnd b3 ,
                                                    Bnd b4 ) : Array(label,data,Bnds(b1,b2,b3,b4)) {
      static_assert( rank == 4 , "ERROR: Calling constructor with 4 bound on non-rank-4 array" );
    }
    /** @brief 5-D non-owned constructor
      * \copydetails doxhide_FArray_non_owned_constructors */
    YAKL_INLINE Array( char const *label , T *data, Bnd b1 ,
                                                    Bnd b2 ,
                                                    Bnd b3 ,
                                                    Bnd b4 ,
                                                    Bnd b5 ) : Array(label,data,Bnds(b1,b2,b3,b4,b5)) {
      static_assert( rank == 5 , "ERROR: Calling constructor with 5 bound on non-rank-5 array" );
    }
    /** @brief 6-D non-owned constructor
      * \copydetails doxhide_FArray_non_owned_constructors */
    YAKL_INLINE Array( char const *label , T *data, Bnd b1 ,
                                                    Bnd b2 ,
                                                    Bnd b3 ,
                                                    Bnd b4 ,
                                                    Bnd b5 ,
                                                    Bnd b6 ) : Array(label,data,Bnds(b1,b2,b3,b4,b5,b6)) {
      static_assert( rank == 6 , "ERROR: Calling constructor with 6 bound on non-rank-6 array" );
    }
    /** @brief 7-D non-owned constructor
      * \copydetails doxhide_FArray_non_owned_constructors */
    YAKL_INLINE Array( char const *label , T *data, Bnd b1 ,
                                                    Bnd b2 ,
                                                    Bnd b3 ,
                                                    Bnd b4 ,
                                                    Bnd b5 ,
                                                    Bnd b6 ,
                                                    Bnd b7 ) : Array(label,data,Bnds(b1,b2,b3,b4,b5,b6,b7)) {
      static_assert( rank == 7 , "ERROR: Calling constructor with 7 bound on non-rank-7 array" );
    }
    /** @brief 8-D non-owned constructor
      * \copydetails doxhide_FArray_non_owned_constructors */
    YAKL_INLINE Array( char const *label , T *data, Bnd b1 ,
                                                    Bnd b2 ,
                                                    Bnd b3 ,
                                                    Bnd b4 ,
                                                    Bnd b5 ,
                                                    Bnd b6 ,
                                                    Bnd b7 ,
                                                    Bnd b8 ) : Array(label,data,Bnds(b1,b2,b3,b4,b5,b6,b7,b8)) {
      static_assert( rank == 8 , "ERROR: Calling constructor with 8 bound on non-rank-8 array" );
    }
    /** @brief Generic initializer-list or std::vector based owned constructor
      * \copydetails doxhide_FArray_non_owned_constructors */
    YAKL_INLINE Array(char const *label, T *data, Bnds bnds) {
      static_assert( rank >= 1 && rank <= 8 , "ERROR: Creating Array with a rank < 1 or > 8" );
      nullify();
      #ifdef YAKL_DEBUG
        if ( bnds.size() < rank ) { yakl_throw("ERROR: Number of array bounds specified is < rank"); }
        if (data == nullptr) yakl_throw("ERROR: wrapping nullptr with a YAKL Array object");
      #endif
      this->myname = label;
      for (int i=0; i < rank; i++) { this->lbounds[i] = bnds[i].l; this->dimension[i] = bnds[i].u - bnds[i].l + 1; }
      this->myData = data;
      this->refCount = nullptr;
    }


    /*
    COPY CONSTRUCTORS / FUNCTIONS
    This shares the pointers with another Array and increments the refCounter
    */
    /** @brief Copy metadata, share data pointer; if owned, increment reference counter. No deep copy. */
    YAKL_INLINE Array(Array<non_const_value_type,rank,myMem,styleFortran> const &rhs) {
      // This is a constructor, so no need to deallocate
      nullify();
      copy_constructor_common(rhs);
    }
    /** @brief Copy metadata, share data pointer; if owned, increment reference counter. No deep copy. */
    YAKL_INLINE Array(Array<const_value_type,rank,myMem,styleFortran> const &rhs) {
      static_assert( std::is_const<T>::value , 
                     "ERROR: Cannot create non-const Array using const Array" );
      // This is a constructor, so no need to deallocate
      nullify();
      copy_constructor_common(rhs);
    }


    /** @brief Copy metadata, share data pointer; if owned, increment reference counter. No deep copy. */
    YAKL_INLINE Array & operator=(Array<non_const_value_type,rank,myMem,styleFortran> const &rhs) {
      if constexpr (! std::is_const<T>::value) {
        if (this == &rhs) { return *this; }
      }
      YAKL_EXECUTE_ON_HOST_ONLY( this->deallocate(); )
      copy_constructor_common(rhs);
      return *this;
    }
    /** @brief Copy metadata, share data pointer; if owned, increment reference counter. No deep copy. */
    YAKL_INLINE Array & operator=(Array<const_value_type,rank,myMem,styleFortran> const &rhs) {
      static_assert( std::is_const<T>::value , 
                     "ERROR: Cannot create non-const Array using const Array" );
      if constexpr (std::is_const<T>::value) {
        if (this == &rhs) { return *this; }
      }
      YAKL_EXECUTE_ON_HOST_ONLY( this->deallocate(); )
      copy_constructor_common(rhs);
      return *this;
    }

    /** @private */
    template <class TLOC>
    YAKL_INLINE void copy_constructor_common(Array<TLOC,rank,myMem,styleFortran> const &rhs) {
      for (int i=0; i<rank; i++) {
        this->lbounds[i] = rhs.lbounds[i]; this->dimension[i] = rhs.dimension[i];
      }
      this->myname = rhs.myname;
      this->myData   = rhs.myData;
      YAKL_EXECUTE_ON_HOST_ONLY( yakl_mtx_lock(); )
      this->refCount = rhs.refCount;
      if (this->refCount != nullptr) {
        // YAKL_EXECUTE_ON_HOST_ONLY( (*(this->refCount))++; )  // This gives an nvc++ error
        YAKL_EXECUTE_ON_HOST_ONLY( { (*(this->refCount))++; } )  // This works around the nvc++ error
      }
      YAKL_EXECUTE_ON_HOST_ONLY( yakl_mtx_unlock(); )
    }


    /*
    MOVE CONSTRUCTORS
    This steals the pointers form the rhs instead of sharing and sets rhs pointers to nullptr.
    Therefore, no need to increment reference counter
    */
    /** @brief Move metadata and data pointer. No deep copy. */
    YAKL_INLINE Array(Array &&rhs) {
      // This is a constructor, so no need to deallocate
      nullify();
      for (int i=0; i<rank; i++) {
        this->lbounds[i] = rhs.lbounds[i]; this->dimension[i] = rhs.dimension[i];
      }
      this->myname = rhs.myname;
      this->myData   = rhs.myData;
      rhs.myData   = nullptr;

      this->refCount = rhs.refCount;
      rhs.refCount = nullptr;
    }


    /** @brief Move metadata and data pointer. No deep copy. */
    YAKL_INLINE Array& operator=(Array &&rhs) {
      if (this == &rhs) { return *this; }
      YAKL_EXECUTE_ON_HOST_ONLY( this->deallocate(); )
      for (int i=0; i<rank; i++) {
        this->lbounds  [i] = rhs.lbounds  [i]; this->dimension[i] = rhs.dimension[i];
      }
      this->myname = rhs.myname;
      this->myData   = rhs.myData;
      rhs.myData   = nullptr;

      this->refCount = rhs.refCount;
      rhs.refCount = nullptr;

      return *this;
    }


    /*
    DESTRUCTOR
    Decrement the refCounter, and if it's zero, deallocate and nullify.  
    */
    /** @brief If owned, decrement reference counter, and deallocate data when it reaches zero. If non-owned, does nothing */
    YAKL_INLINE ~Array() {
      YAKL_EXECUTE_ON_HOST_ONLY( this->deallocate(); )
    }


    /** @brief Construct this FArray object from an ArrayIR object for easy interoperability with other C++ portability libraries
      * 
      * @param  ir            The ArrayIR object
      * @param  lower_bounds  The lower bounds to use for this FArray object (optional) */
    Array( array_ir::ArrayIR<T,rank> const &ir , std::vector<int> lower_bounds = std::vector<int>() ) {
      nullify();
      if (myMem == memDevice && (! ir.data_valid_on_device())) yakl_throw("ERROR: wrapping non-device-valid ArrayIR with memDevice yakl::FArray");
      if (myMem == memHost   && (! ir.data_valid_on_host  ())) yakl_throw("ERROR: wrapping non-host-valid ArrayIR with memHost yakl::FArray");
      this->myData = ir.data();
      this->myname = ir.label();
      for (int i=0; i < rank; i++) { this->dimension[i] = ir.extent(rank-1-i); }
      if ( (! lower_bounds.empty()) && ( lower_bounds.size() != rank ) ) yakl_throw("ERROR: Passed lower bounds of the wrong rank");
      for (int i=0; i < rank; i++) { this->lbounds[i] = lower_bounds[i]; }
    }


    /** @brief Create an ArrayIR object from this FArray object for easy interoperability with other C++ portability libraries.
      *        Lower bounds are discarded. */
    template <class TLOC = T>
    array_ir::ArrayIR<TLOC,rank> create_ArrayIR() const {
      std::array<size_t,rank> dimensions;
      for (int i=0; i < rank; i++) { dimensions[i] = this->dimension[rank-1-i]; }
      if (myMem == memHost) {
        return array_ir::ArrayIR<TLOC,rank>(const_cast<TLOC *>(this->myData),dimensions,array_ir::MEMORY_HOST,this->label());
      } else {
        #ifdef YAKL_MANAGED_MEMORY
          return array_ir::ArrayIR<TLOC,rank>(const_cast<TLOC *>(this->myData),dimensions,array_ir::MEMORY_SHARED,this->label());
        #else
          return array_ir::ArrayIR<TLOC,rank>(const_cast<TLOC *>(this->myData),dimensions,array_ir::MEMORY_DEVICE,this->label());
        #endif
      }
    }


    // Common detailed documentation for all indexers
    /** @class doxhide_FArray_indexers
      * @brief dummy
      * 
      * Return a reference to the element at the requested index. Number of indices must match the number of dimensions
      * in this array, `N`. The array object must already be allocated. For index checking, please define the `YAKL_DEBUG`
      * CPP macro. Use one-based indexing (modifying lower bounds for non-standard lower bound arrays) with the left-most
      * value varying the fastest (column-major ordering).
      */
    /* ARRAY INDEXERS (FORTRAN index ordering)
    Return the element at the given index (either read-only or read-write)
    */
    /** @brief Return reference to element at the requested index (1-D)
      * \copydetails doxhide_FArray_indexers */
    YAKL_INLINE T &operator()(int i0) const {
      static_assert( rank == 1 , "ERROR: Indexing non-rank-1 array with 1 index" );
      #ifdef YAKL_DEBUG
        check(i0);
      #endif
      index_t ind = (i0-this->lbounds[0]);
      return this->myData[ind];
    }
    /** @brief Return reference to element at the requested index (2-D)
      * \copydetails doxhide_FArray_indexers */
    YAKL_INLINE T &operator()(int i0, int i1) const {
      static_assert( rank == 2 , "ERROR: Indexing non-rank-2 array with 2 indices" );
      #ifdef YAKL_DEBUG
        check(i0,i1);
      #endif
      index_t ind =                       (i1-this->lbounds[1])  *
                     this->dimension[0] + (i0-this->lbounds[0]) ;
      return this->myData[ind];
    }
    /** @brief Return reference to element at the requested index (3-D)
      * \copydetails doxhide_FArray_indexers */
    YAKL_INLINE T &operator()(int i0, int i1, int i2) const {
      static_assert( rank == 3 , "ERROR: Indexing non-rank-3 array with 3 indices" );
      #ifdef YAKL_DEBUG
        check(i0,i1,i2);
      #endif
      index_t ind = (                      (i2-this->lbounds[2])  *
                      this->dimension[1] + (i1-this->lbounds[1]) )*
                      this->dimension[0] + (i0-this->lbounds[0]) ;
      return this->myData[ind];
    }
    /** @brief Return reference to element at the requested index (4-D)
      * \copydetails doxhide_FArray_indexers */
    YAKL_INLINE T &operator()(int i0, int i1, int i2, int i3) const {
      static_assert( rank == 4 , "ERROR: Indexing non-rank-4 array with 4 indices" );
      #ifdef YAKL_DEBUG
        check(i0,i1,i2,i3);
      #endif
      index_t ind = ((                      (i3-this->lbounds[3])  *
                       this->dimension[2] + (i2-this->lbounds[2]) )*
                       this->dimension[1] + (i1-this->lbounds[1]) )*
                       this->dimension[0] + (i0-this->lbounds[0]) ;
      return this->myData[ind];
    }
    /** @brief Return reference to element at the requested index (5-D)
      * \copydetails doxhide_FArray_indexers */
    YAKL_INLINE T &operator()(int i0, int i1, int i2, int i3, int i4) const {
      static_assert( rank == 5 , "ERROR: Indexing non-rank-5 array with 5 indices" );
      #ifdef YAKL_DEBUG
        check(i0,i1,i2,i3,i4);
      #endif
      index_t ind = (((                      (i4-this->lbounds[4])  *
                        this->dimension[3] + (i3-this->lbounds[3]) )*
                        this->dimension[2] + (i2-this->lbounds[2]) )*
                        this->dimension[1] + (i1-this->lbounds[1]) )*
                        this->dimension[0] + (i0-this->lbounds[0]) ;
      return this->myData[ind];
    }
    /** @brief Return reference to element at the requested index (6-D)
      * \copydetails doxhide_FArray_indexers */
    YAKL_INLINE T &operator()(int i0, int i1, int i2, int i3, int i4, int i5) const {
      static_assert( rank == 6 , "ERROR: Indexing non-rank-6 array with 6 indices" );
      #ifdef YAKL_DEBUG
        check(i0,i1,i2,i3,i4,i5);
      #endif
      index_t ind = ((((                      (i5-this->lbounds[5])  *
                         this->dimension[4] + (i4-this->lbounds[4]) )*
                         this->dimension[3] + (i3-this->lbounds[3]) )*
                         this->dimension[2] + (i2-this->lbounds[2]) )*
                         this->dimension[1] + (i1-this->lbounds[1]) )*
                         this->dimension[0] + (i0-this->lbounds[0]) ;
      return this->myData[ind];
    }
    /** @brief Return reference to element at the requested index (7-D)
      * \copydetails doxhide_FArray_indexers */
    YAKL_INLINE T &operator()(int i0, int i1, int i2, int i3, int i4, int i5, int i6) const {
      static_assert( rank == 7 , "ERROR: Indexing non-rank-7 array with 7 indices" );
      #ifdef YAKL_DEBUG
        check(i0,i1,i2,i3,i4,i5,i6);
      #endif
      index_t ind = (((((                      (i6-this->lbounds[6])  *
                          this->dimension[5] + (i5-this->lbounds[5]) )*
                          this->dimension[4] + (i4-this->lbounds[4]) )*
                          this->dimension[3] + (i3-this->lbounds[3]) )*
                          this->dimension[2] + (i2-this->lbounds[2]) )*
                          this->dimension[1] + (i1-this->lbounds[1]) )*
                          this->dimension[0] + (i0-this->lbounds[0]) ;
      return this->myData[ind];
    }
    /** @brief Return reference to element at the requested index (8-D)
      * \copydetails doxhide_FArray_indexers */
    YAKL_INLINE T &operator()(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7) const {
      static_assert( rank == 8 , "ERROR: Indexing non-rank-8 array with 8 indices" );
      #ifdef YAKL_DEBUG
        check(i0,i1,i2,i3,i4,i5,i6,i7);
      #endif
      index_t ind = ((((((                      (i7-this->lbounds[7])  *
                           this->dimension[6] + (i6-this->lbounds[6]) )*
                           this->dimension[5] + (i5-this->lbounds[5]) )*
                           this->dimension[4] + (i4-this->lbounds[4]) )*
                           this->dimension[3] + (i3-this->lbounds[3]) )*
                           this->dimension[2] + (i2-this->lbounds[2]) )*
                           this->dimension[1] + (i1-this->lbounds[1]) )*
                           this->dimension[0] + (i0-this->lbounds[0]) ;
      return this->myData[ind];
    }

    
    /** @private */
    YAKL_INLINE void check(int i0, int i1=1, int i2=1, int i3=1, int i4=1, int i5=1,
                           int i6=1, int i7=1) const {
      #ifdef YAKL_DEBUG
        if (! this->initialized()) { yakl_throw("Error: Using operator() on an Array that isn't allocated"); }
        if constexpr (rank >= 1) { if (i0 < this->lbounds[0] || i0 >= this->lbounds[0]+this->dimension[0]) ind_out_bounds<0>(i0); }
        if constexpr (rank >= 2) { if (i1 < this->lbounds[1] || i1 >= this->lbounds[1]+this->dimension[1]) ind_out_bounds<1>(i1); }
        if constexpr (rank >= 3) { if (i2 < this->lbounds[2] || i2 >= this->lbounds[2]+this->dimension[2]) ind_out_bounds<2>(i2); }
        if constexpr (rank >= 4) { if (i3 < this->lbounds[3] || i3 >= this->lbounds[3]+this->dimension[3]) ind_out_bounds<3>(i3); }
        if constexpr (rank >= 5) { if (i4 < this->lbounds[4] || i4 >= this->lbounds[4]+this->dimension[4]) ind_out_bounds<4>(i4); }
        if constexpr (rank >= 6) { if (i5 < this->lbounds[5] || i5 >= this->lbounds[5]+this->dimension[5]) ind_out_bounds<5>(i5); }
        if constexpr (rank >= 7) { if (i6 < this->lbounds[6] || i6 >= this->lbounds[6]+this->dimension[6]) ind_out_bounds<6>(i6); }
        if constexpr (rank >= 8) { if (i7 < this->lbounds[7] || i7 >= this->lbounds[7]+this->dimension[7]) ind_out_bounds<7>(i7); }
        #if defined(YAKL_SEPARATE_MEMORY_SPACE)
          YAKL_EXECUTE_ON_DEVICE_ONLY( if constexpr (myMem == memHost) yakl_throw("ERROR: host array being accessed in a device kernel"); )
          #if !defined(YAKL_MANAGED_MEMORY)
            YAKL_EXECUTE_ON_HOST_ONLY(
              if constexpr (myMem == memDevice) {
                std::cerr << "ERROR: For Array labeled: " << this->myname << ":" << std::endl;
                yakl_throw("Device array being accessed on the host without managed memory turned on");
              }
            )
          #endif
        #endif
      #endif
    }


    // if this function gets called, then there was definitely an error
    /** @private */
    template <int I>
    YAKL_INLINE void ind_out_bounds(int ind) const {
      #ifdef YAKL_DEBUG
        YAKL_EXECUTE_ON_HOST_ONLY(
          std::cerr << "ERROR: For Array labeled: " << this->myname << ":" << std::endl;
          std::cerr << "Index " << I+1 << " of " << rank << " is out of bounds.  Provided index: " << ind
                    << ".  Lower Bound: " << this->lbounds[I] << ".  Upper Bound: " << this->dimension[I]-1 << std::endl;
        )
        yakl_throw("ERROR: Index out of bounds.");
      #endif
    }


    /** @brief [ASYNCHRONOUS] Assign a scalar arithmetic value to all entries in this array object */
    template <class TLOC, typename std::enable_if<std::is_arithmetic<TLOC>::value,bool>::type = false>
    Array operator=(TLOC const &rhs) const {
      memset_loc(rhs);
      return *this;
    }


    /** @private */
    template <class TLOC>
    void memset_loc(TLOC rhs) const {
      if (myMem == memDevice) {
        #ifdef YAKL_ENABLE_STREAMS
          fence();
        #endif
        memset(*this, rhs);
        #ifdef YAKL_ENABLE_STREAMS
          fence();
        #endif
      } else {
        for (int i=0; i < this->totElems(); i++) { this->myData[i] = rhs; }
      }
    }


    /** @brief Return an array aliasing a contiguous subset of the slowest dimension. Retuns the same array with the slowest dimension truncated
               to [lbounds(rank-1) , u]. */
    YAKL_INLINE Array<T,rank,myMem,styleFortran> subset_slowest_dimension(int u) const { return subset_slowest_dimension(lbounds[rank-1],u); }


    /** @brief Return an array aliasing a contiguous subset of the slowest dimension */
    YAKL_INLINE Array<T,rank,myMem,styleFortran> subset_slowest_dimension(int l, int u) const {
      #ifdef YAKL_DEBUG
        if (! this->initialized()) { yakl_throw("ERROR: Trying to subset_slowest_dimension an Array that hasn't been initialized"); }
        if (l < lbounds[rank-1]) { yakl_throw("ERROR: subset_slowest_dimension lower bound too low"); }
        if (u < lbounds[rank-1]) { yakl_throw("ERROR: subset_slowest_dimension upper bound too low"); }
        if (l > u) { yakl_throw("ERROR: subset_slowest_dimension lower bound > upper bounds"); }
        if (u >= this->lbounds[rank-1]+this->dimension[rank-1]) { yakl_throw("ERROR: subset_slowest_dimension upper bound too high"); }
      #endif
      auto ret = *this;
      auto &lb = this->lbounds;
      auto &d  = this->dimension;
      if constexpr (rank == 1) ret.myData += (l-lb[rank-1]);
      if constexpr (rank == 2) ret.myData += (l-lb[rank-1])*d[0];
      if constexpr (rank == 3) ret.myData += (l-lb[rank-1])*d[0]*d[1];
      if constexpr (rank == 4) ret.myData += (l-lb[rank-1])*d[0]*d[1]*d[2];
      if constexpr (rank == 5) ret.myData += (l-lb[rank-1])*d[0]*d[1]*d[2]*d[3];
      if constexpr (rank == 6) ret.myData += (l-lb[rank-1])*d[0]*d[1]*d[2]*d[3]*d[4];
      if constexpr (rank == 7) ret.myData += (l-lb[rank-1])*d[0]*d[1]*d[2]*d[3]*d[4]*d[5];
      if constexpr (rank == 8) ret.myData += (l-lb[rank-1])*d[0]*d[1]*d[2]*d[3]*d[4]*d[5]*d[6];
      ret.dimension[rank-1] = u-l+1;
      return ret;
    }


    /** @class doxhide_FArray_slicing
      * @brief dummy
      *
      * Returns an array object with the requested slice. Slices must be contiguous, and only whole dimensions
      * may be sliced. Please use yakl::COLON to denote dimensions being sliced to make this more clear.
      * The template parameter specifies the number of dimensions in the resulting sliced array object.
      * The array object being sliced must be already allocated, and the number of sliced dimensions must
      * not exceed the rank of the array being sliced. If slicing is performed on the host, then the returned
      * array object is owned and reference counted, guaranteeing the underling data remains valid while the returned
      * slice is used. If slicing is performed on the device, a non-owned array object is returned, though this
      * rarely if ever presents an issue on the device.
      * 
      * Example usage:
      * ```
      * auto myslice = arr.slice<2>({COLON,COLON,k});
      * auto myslice = arr.slice<1>(COLON,i4,i3,i2,i1);
      * ```
      * @param dims yakl::Dims object specifying the indices at which the slice should occur as well as the dimensions
      *        that should be sliced.
      * @param i[0-7]: Index of the array slice.
      */

    /** @brief Array slice using initializer list or std::vector indices 
      * \copydetails doxhide_FArray_slicing */
    template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( Dims const &dims ) const {
      #ifdef YAKL_DEBUG
        if (rank != dims.size()) {
          YAKL_EXECUTE_ON_HOST_ONLY( std::cerr << "For Array named " << this->myname << ":  "; )
          yakl_throw("ERROR: rank must be equal to dims.size()");
        }
        for (int i=N; i<rank; i++) {
          if (dims.data[i] < this->lbounds[i] || dims.data[i] >= this->lbounds[i]+this->dimension[i] ) {
            YAKL_EXECUTE_ON_HOST_ONLY( std::cerr << "For Array named " << this->myname << ":  "; )
            yakl_throw("ERROR: One of the slicing dimension dimensions is out of bounds");
          }
        }
        if (! this->initialized()) {
          YAKL_EXECUTE_ON_HOST_ONLY( std::cerr << "For Array named " << this->myname << ":  "; )
          yakl_throw("ERROR: calling slice() on an Array that hasn't been allocated");
        }
      #endif
      Array<T,N,myMem,styleFortran> ret;
      index_t offset = 1;
      for (int i=0; i<N; i++) {
        ret.dimension[i] = this->dimension[i];
        ret.lbounds  [i] = this->lbounds  [i];
        offset *= this->dimension[i];
      }
      index_t retOff = 0;
      for (int i=N; i<rank; i++) {
        retOff += (dims.data[i]-this->lbounds[i])*offset;
        offset *= this->dimension[i];
      }
      ret.myData = &(this->myData[retOff]);
      YAKL_EXECUTE_ON_HOST_ONLY(
        yakl_mtx_lock();
        ret.refCount = this->refCount;
        if (this->refCount != nullptr) {
          (*(this->refCount))++;
        }
        yakl_mtx_unlock();
      )
      return ret;
    }
    /** @brief Array slice of 1-D array 
      * \copydetails doxhide_FArray_slicing */
    template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0 ) const {
      static_assert( rank == 1 , "ERROR: Calling slice() with 1 index on a non-rank-1 array" );
      static_assert( N <= rank , "ERROR: Calling slice() with more dimenions than this array's rank" );
      return slice<N>( Dims(i0) );
    }
    /** @brief Array slice of 2-D array 
      * \copydetails doxhide_FArray_slicing */
    template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0, int i1 ) const {
      static_assert( rank == 2 , "ERROR: Calling slice() with 2 index on a non-rank-2 array" );
      static_assert( N <= rank , "ERROR: Calling slice() with more dimenions than this array's rank" );
      return slice<N>( Dims(i0,i1) );
    }
    /** @brief Array slice of 3-D array 
      * \copydetails doxhide_FArray_slicing */
    template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0, int i1, int i2 ) const {
      static_assert( rank == 3 , "ERROR: Calling slice() with 3 index on a non-rank-3 array" );
      static_assert( N <= rank , "ERROR: Calling slice() with more dimenions than this array's rank" );
      return slice<N>( Dims(i0,i1,i2) );
    }
    /** @brief Array slice of 4-D array 
      * \copydetails doxhide_FArray_slicing */
    template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0, int i1, int i2, int i3 ) const {
      static_assert( rank == 4 , "ERROR: Calling slice() with 4 index on a non-rank-4 array" );
      static_assert( N <= rank , "ERROR: Calling slice() with more dimenions than this array's rank" );
      return slice<N>( Dims(i0,i1,i2,i3) );
    }
    /** @brief Array slice of 5-D array 
      * \copydetails doxhide_FArray_slicing */
    template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0, int i1, int i2, int i3,
                                                                      int i4 ) const {
      static_assert( rank == 5 , "ERROR: Calling slice() with 5 index on a non-rank-5 array" );
      static_assert( N <= rank , "ERROR: Calling slice() with more dimenions than this array's rank" );
      return slice<N>( Dims(i0,i1,i2,i3,i4) );
    }
    /** @brief Array slice of 6-D array 
      * \copydetails doxhide_FArray_slicing */
    template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0, int i1, int i2, int i3,
                                                                      int i4, int i5 ) const {
      static_assert( rank == 6 , "ERROR: Calling slice() with 6 index on a non-rank-6 array" );
      static_assert( N <= rank , "ERROR: Calling slice() with more dimenions than this array's rank" );
      return slice<N>( Dims(i0,i1,i2,i3,i4,i5) );
    }
    /** @brief Array slice of 7-D array 
      * \copydetails doxhide_FArray_slicing */
    template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0, int i1, int i2, int i3,
                                                                      int i4, int i5, int i6 ) const {
      static_assert( rank == 7 , "ERROR: Calling slice() with 7 index on a non-rank-7 array" );
      static_assert( N <= rank , "ERROR: Calling slice() with more dimenions than this array's rank" );
      return slice<N>( Dims(i0,i1,i2,i3,i4,i5,i6) );
    }
    /** @brief Array slice of 8-D array 
      * \copydetails doxhide_FArray_slicing */
    template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> slice( int i0, int i1, int i2, int i3,
                                                                      int i4, int i5, int i6, int i7 ) const {
      static_assert( rank == 8 , "ERROR: Calling slice() with 8 index on a non-rank-8 array" );
      static_assert( N <= rank , "ERROR: Calling slice() with more dimenions than this array's rank" );
      return slice<N>( Dims(i0,i1,i2,i3,i4,i5,i6,i7) );
    }



    /** @class doxhide_FArray_reshape
      * @brief dummy
      *
      * Returns an array object that shares the data pointer of this array object but has different dimensions,
      * specified by the passed yakl::Dims object (or initializer list of integers). The total number of array elements
      * must remain the same, the memory space must remain the same, the style must remain the same, and the array must
      * already be allocated. This is a fast operation. No allocations are performed, and no underlying data is allocated.
      * If this is performed on the host, then the returned array is owned, and the data pointer's reference
      * counter is incremented. This means you're guaranteed the data pointer is valid throughout the use
      * of the returned array object. If this is performed on the device, then the returned array is non-owned.
      * Be careful doing this in the innermost loop, even on the host, though, because it is still copying
      * array metadata, and you may notice the extra cost. 
      *
      * Example usage:
      * ```
      * auto new_arr = arr.reshape({ny,nx});
      * auto new_arr = arr.reshape(nx,{1,ny+1},nz);
      * auto new_arr = arr.reshape(n);
      * ```
      * @param bnds   yakl::Bnds object containing the bounds of each dimension of the returned array object
      * @param i[0-7] dimensions of the newly reshaped array
      */

    /** @brief Reshape array using initializer list or std::vector indices
      * \copydetails doxhide_FArray_reshape */
    template <int N> YAKL_INLINE Array<T,N,myMem,styleFortran> reshape(Bnds const &bnds) const {
      #ifdef YAKL_DEBUG
        if (! this->initialized()) { yakl_throw("ERROR: Trying to reshape an Array that hasn't been initialized"); }
        if (bnds.size() != N) { yakl_throw("ERROR: new number of reshaped array dimensions does not match the templated rank"); }
        index_t totelems = 1;
        for (int i=0; i < N; i++) { totelems *= (bnds.u[i]-bnds.l[i]+1); }
        if (totelems != this->totElems()) { yakl_throw("ERROR: Total number of reshaped array elements is not consistent with this array"); }
      #endif
      Array<T,N,myMem,styleFortran> ret;
      for (int i=0; i < N; i++) {
        ret.dimension[i] = bnds.u[i] - bnds.l[i] + 1;  ret.lbounds  [i] = bnds.l[i];
      }
      ret.myname = this->myname;
      ret.myData = this->myData;
      YAKL_EXECUTE_ON_HOST_ONLY(
        yakl_mtx_lock();
        ret.refCount = this->refCount;
        if (this->refCount != nullptr) {
          (*(this->refCount))++;
        }
        yakl_mtx_unlock();
      )
      return ret;
    }
    /** @brief Reshape array into a 1-D array
      * \copydetails doxhide_FArray_reshape */
    YAKL_INLINE Array<T,1,myMem,styleFortran> reshape(Bnd b0                                                        ) const { return reshape<1>( Bnds(b0) ); }
    /** @brief Reshape array into a 2-D array
      * \copydetails doxhide_FArray_reshape */
    YAKL_INLINE Array<T,2,myMem,styleFortran> reshape(Bnd b0, Bnd b1                                                ) const { return reshape<2>( Bnds(b0,b1) ); }
    /** @brief Reshape array into a 3-D array
      * \copydetails doxhide_FArray_reshape */
    YAKL_INLINE Array<T,3,myMem,styleFortran> reshape(Bnd b0, Bnd b1, Bnd b2                                        ) const { return reshape<3>( Bnds(b0,b1,b2) ); }
    /** @brief Reshape array into a 4-D array
      * \copydetails doxhide_FArray_reshape */
    YAKL_INLINE Array<T,4,myMem,styleFortran> reshape(Bnd b0, Bnd b1, Bnd b2, Bnd b3                                ) const { return reshape<4>( Bnds(b0,b1,b2,b3) ); }
    /** @brief Reshape array into a 5-D array
      * \copydetails doxhide_FArray_reshape */
    YAKL_INLINE Array<T,5,myMem,styleFortran> reshape(Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4                        ) const { return reshape<5>( Bnds(b0,b1,b2,b3,b4) ); }
    /** @brief Reshape array into a 6-D array
      * \copydetails doxhide_FArray_reshape */
    YAKL_INLINE Array<T,6,myMem,styleFortran> reshape(Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5                ) const { return reshape<6>( Bnds(b0,b1,b2,b3,b4,b5) ); }
    /** @brief Reshape array into a 7-D array
      * \copydetails doxhide_FArray_reshape */
    YAKL_INLINE Array<T,7,myMem,styleFortran> reshape(Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5, Bnd b6        ) const { return reshape<7>( Bnds(b0,b1,b2,b3,b4,b5,b6) ); }
    /** @brief Reshape array into a 8-D array
      * \copydetails doxhide_FArray_reshape */
    YAKL_INLINE Array<T,8,myMem,styleFortran> reshape(Bnd b0, Bnd b1, Bnd b2, Bnd b3, Bnd b4, Bnd b5, Bnd b6, Bnd b7) const { return reshape<8>( Bnds(b0,b1,b2,b3,b4,b5,b6,b7) ); }


    /** @brief Collapse this array into a 1-D array
      *
      * Returns an array object that shares the data pointer of this array object but has only one dimension,
      * with all of this array object's dimensions collapsed into a single dimension. 
      * This is a fast operation. No allocations are performed, and no underlying data is allocated.
      * If this is performed on the host, then the returned array is owned, and the data pointer's reference
      * counter is incremented. This means you're guaranteed the data pointer is valid throughout the use
      * of the returned array object. If this is performed on the device, then the returned array is non-owned.
      * Be careful doing this in the innermost loop, even on the host, though, because it is still copying
      * array metadata, and you may notice the extra cost. 
      * Example usage: `auto new_arr_1d = arr.collapse();`
      */
    YAKL_INLINE Array<T,1,myMem,styleFortran> collapse(int lbnd=1) const {
      #ifdef YAKL_DEBUG
        if (! this->initialized()) { yakl_throw("ERROR: Trying to collapse an Array that hasn't been initialized"); }
      #endif
      Array<T,1,myMem,styleFortran> ret;
      ret.dimension[0] = this->totElems();  ret.lbounds  [0] = lbnd;
      ret.myname = this->myname;
      ret.myData = this->myData;
      YAKL_EXECUTE_ON_HOST_ONLY(
        yakl_mtx_lock();
        ret.refCount = this->refCount;
        if (this->refCount != nullptr) {
          (*(this->refCount))++;
        }
        yakl_mtx_unlock();
      )
      return ret;
    }


    // Create a host copy of this array. Even if the array exists on the host, a deep copy to a separate
    // object is still performed to avoid any potential bugs when the user expects this behavior
    /** @brief [DEEP_COPY] Create a copy of this array in yakl::memHost space
      * 
      * Create and allocate a yakl::memHost array object of the same type, rank, dimensions, and style. Then deep copy
      * the data from this array object to the array object returned by this function. This is a slow routine.
      * It both allocates and deep copies the underlying data.
      * 
      * Even if the current array is yakl::memHost, this will still allocate and copy to a new object.
      * 
      * @returns A newly allocated array of the same type, rank, and style as this one in yakl::memHost space with
      *          data copied from this array object.
      */
    template <class TLOC=T>
    inline Array<TLOC,rank,memHost,styleFortran> createHostCopy(Stream stream = Stream()) const {
      auto ret = createHostObject();
      this->copy_inform(ret);
      if (myMem == memHost) { memcpy_host_to_host  ( ret.myData , this->myData , this->totElems()          ); }
      else                  { memcpy_device_to_host( ret.myData , this->myData , this->totElems() , stream ); }
      if (stream.is_default_stream()) { fence(); }
      else                            { stream.fence(); }
      return Array<TLOC,rank,memHost,styleFortran>(ret);
    }


    // Create a separate host Array with the same rank memory space and style
    /** @brief Create and allocate a yakl::memHost array object of the same type, rank, dimensions, and style.
      * 
      * This is the same as createHostCopy() but without the data deep copy portion.
      * This may be slow since host objects do not use the YAKL pool allocator.
      * NOTE: This does not deep copy data. It merely creates and allocates a new array object and returns it.
      * NOTE: The returned array will have a **non-`const`** underlying type. */
    template <class TLOC=typename std::remove_cv<T>::type>
    inline Array<typename std::remove_cv<TLOC>::type,rank,memHost,styleFortran> createHostObject() const {
      #ifdef YAKL_DEBUG
        if (! this->initialized()) {
          YAKL_EXECUTE_ON_HOST_ONLY( std::cerr << "For Array named " << this->myname << ":  "; )
          yakl_throw("Error: createHostCopy() called on an Array that hasn't been allocated.");
        }
      #endif
      // If this Array is of const type, then we need to use non-const when allocating, then cast it to const aterward
      Array<typename std::remove_cv<TLOC>::type,rank,memHost,styleFortran> ret;  // nullified + owned == true
      for (int i=0; i<rank; i++) { ret.lbounds[i] = this->lbounds[i];  ret.dimension[i] = this->dimension[i]; }
      ret.myname = this->myname;
      ret.allocate();
      return ret;
    }


    // Create a device copy of this array. Even if the array exists on the host, a deep copy to a separate
    // object is still performed to avoid any potential bugs when the user expects this behavior
    /** @brief [DEEP_COPY] Create a copy of this array in yakl::memDevice space
      *
      * Create and allocate a yakl::memDevice array object of the same type, rank, dimensions, and style. Then deep copy
      * the data from this array object to the array object returned by this function. This is a slow routine.
      * It both allocates and deep copies the underlying data.
      * 
      * Even if the current array is yakl::memDevice, this will still allocate and copy to a new object.
      * 
      * @returns A newly allocated array of the same type, rank, and style as this one in yakl::memDevice space with
      *          data copied from this array object.
      */
    template <class TLOC=T>
    inline Array<TLOC,rank,memDevice,styleFortran> createDeviceCopy(Stream stream = Stream()) const {
      auto ret = createDeviceObject();
      this->copy_inform(ret);
      if (myMem == memHost) { memcpy_host_to_device  ( ret.myData , this->myData , this->totElems() , stream ); }
      else                  { memcpy_device_to_device( ret.myData , this->myData , this->totElems() , stream ); }
      if (stream.is_default_stream()) { fence(); }
      else                            { stream.fence(); }
      return Array<TLOC,rank,memDevice,styleFortran>(ret);
    }


    // Create separate device array with the same rank, memory space, and style
    /** @brief Create and allocate a yakl::memDevice array object of the same type, rank, dimensions, and style.
      *
      * This is the same as createDeviceCopy() but without the data deep copy portion.
      * This is a fairly fast routine **if the YAKL pool allocator is enabled**; otherwise, it may be slow.
      * NOTE: This does not deep copy data. It merely creates and allocates a new array object and returns it.
      * NOTE: The returned array will have a **non-`const`** underlying type. */
    template <class TLOC=typename std::remove_cv<T>::type>
    inline Array<typename std::remove_cv<TLOC>::type,rank,memDevice,styleFortran> createDeviceObject() const {
      #ifdef YAKL_DEBUG
        if (! this->initialized()) {
          YAKL_EXECUTE_ON_HOST_ONLY( std::cerr << "For Array named " << this->myname << ":  "; )
          yakl_throw("Error: createHostCopy() called on an Array that hasn't been allocated.");
        }
      #endif
      // If this Array is of const type, then we need to use non-const when allocating, then cast it to const aterward
      Array<typename std::remove_cv<TLOC>::type,rank,memDevice,styleFortran> ret;  // nullified + owned == true
      for (int i=0; i<rank; i++) { ret.lbounds[i] = this->lbounds[i];  ret.dimension[i] = this->dimension[i]; }
      ret.myname = this->myname;
      ret.allocate();
      return ret;
    }


    /* ACCESSORS */
    /** @brief Returns the dimensions of this array as a yakl::FSArray object.
      * 
      * You should use one-based indexing on the returned yakl::FSArray object. */
    YAKL_INLINE FSArray<index_t,1,SB<rank>> get_dimensions() const {
      FSArray<index_t,1,SB<rank>> ret;
      for (int i=0; i<rank; i++) { ret(i+1) = this->dimension[i]; }
      return ret;
    }
    /** @brief Returns the lower bound of each dimension of this array as a yakl::FSArray object.
      * 
      * You should use one-based indexing on the returned yakl::FSArray object. */
    YAKL_INLINE FSArray<int,1,SB<rank>> get_lbounds() const {
      FSArray<int,1,SB<rank>> ret;
      for (int i=0; i<rank; i++) { ret(i+1) = this->lbounds[i]; }
      return ret;
    }
    /** @brief Returns the upper bound of each dimension of this array as a yakl::FSArray object.
      * 
      * You should use one-based indexing on the returned yakl::FSArray object. */
    YAKL_INLINE FSArray<int,1,SB<rank>> get_ubounds() const {
      FSArray<int,1,SB<rank>> ret;
      for (int i=0; i<rank; i++) { ret(i+1) = this->lbounds[i]+this->dimension[i]-1; }
      return ret;
    }
    /** @brief Returns the extent of the requested dimension of this array.
      *
      * **IMPORTANT:** The parameter `dim` is expected to be a **zero-based** index, not a one-based index. */
    YAKL_INLINE index_t extent( int dim ) const {
      #ifdef YAKL_DEBUG
        if (dim < 0 || dim > rank-1) yakl_throw("ERROR: calling extent() with an out of bounds index");
      #endif
      return this->dimension[dim];
    }

  };

}
__YAKL_NAMESPACE_WRAPPER_END__


