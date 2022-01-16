
#pragma once

namespace intrinsics {

  template <class T> YAKL_INLINE int  size(T const &arr, int dim) { return arr.get_dimensions()(dim); }
  template <class T> YAKL_INLINE int  size(T const &arr) { return arr.totElems(); }



  template <class T> YAKL_INLINE auto shape(T const &arr) -> decltype(arr.get_dimensions()) { return arr.get_dimensions(); }



  template <class T> YAKL_INLINE int  lbound (T const &arr, int dim) { return arr.get_lbounds()(dim); }
  template <class T> YAKL_INLINE auto lbound (T const &arr) -> decltype(arr.get_lbounds()) { return arr.get_lbounds(); }



  template <class T> YAKL_INLINE int  ubound (T const &arr, int dim) { return arr.get_ubounds()(dim); }
  template <class T> YAKL_INLINE auto ubound (T const &arr) -> decltype(arr.get_ubounds()) { return arr.get_ubounds(); }



  template <class T> YAKL_INLINE bool allocated (T const &arr) { return arr.myData != nullptr; }



  template <class T> YAKL_INLINE bool associated (T const &arr) { return arr.myData != nullptr; }



  template <class T> YAKL_INLINE T constexpr epsilon(T) { return std::numeric_limits<T>::epsilon(); }
  template <class T, int rank, int myMem, int myStyle>
  YAKL_INLINE T constexpr epsilon(Array<T,rank,myMem,myStyle> const &arr) { return std::numeric_limits<T>::epsilon(); }
  template <class T, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE T constexpr epsilon(FSArray<T,rank,D0,D1,D2,D3> const &arr) { return std::numeric_limits<T>::epsilon(); }
  template <class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE T constexpr epsilon(SArray<T,rank,D0,D1,D2,D3> const &arr) { return std::numeric_limits<T>::epsilon(); }



  template <class T> YAKL_INLINE T constexpr tiny(T) { return std::numeric_limits<T>::min(); }
  template <class T, int rank, int myMem, int myStyle>
  YAKL_INLINE T constexpr tiny(Array<T,rank,myMem,myStyle> const &arr) { return std::numeric_limits<T>::min(); }
  template <class T, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE T constexpr tiny(FSArray<T,rank,D0,D1,D2,D3> const &arr) { return std::numeric_limits<T>::min(); }
  template <class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE T constexpr tiny(SArray<T,rank,D0,D1,D2,D3> const &arr) { return std::numeric_limits<T>::min(); }



  template <class T> YAKL_INLINE T constexpr huge(T) { return std::numeric_limits<T>::max(); }
  template <class T, int rank, int myMem, int myStyle>
  YAKL_INLINE T constexpr huge(Array<T,rank,myMem,myStyle> const &arr) { return std::numeric_limits<T>::max(); }
  template <class T, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE T constexpr huge(FSArray<T,rank,D0,D1,D2,D3> const &arr) { return std::numeric_limits<T>::max(); }
  template <class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE T constexpr huge(SArray<T,rank,D0,D1,D2,D3> const &arr) { return std::numeric_limits<T>::max(); }



  template <class T> YAKL_INLINE T sign(T val) { return val >= 0 ? 1 : -1; }



  template <class T> YAKL_INLINE T mod(T a, T b) { return a - ((int)(a/b) * b); }



  template <class T> YAKL_INLINE T merge(T const t, T const f, bool cond) { return cond ? t : f; }



  ////////////////////////////////////////////////////////////////////////
  // minval
  ////////////////////////////////////////////////////////////////////////
  template <class T, int rank, int myStyle>
  inline T minval( Array<T,rank,memHost,myStyle> const &arr ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling minval on an array that has not been initialized"); }
    #endif
    typename std::remove_cv<T>::type m = arr.myData[0];
    for (int i=1; i<arr.totElems(); i++) {
      if (arr.myData[i] < m) { m = arr.myData[i]; }
    }
    return m;
  }
  template <class T, int rank, int myStyle>
  inline T minval( Array<T,rank,memDevice,myStyle> const &arr ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling minval on an array that has not been initialized"); }
    #endif
    typedef typename std::remove_cv<T>::type TNC; // T Non-Const
    ParallelMin<TNC,memDevice> pmin(arr.totElems());
    return pmin( const_cast<TNC *>(arr.data()) );
  }
  template <class T, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE T minval( FSArray<T,rank,D0,D1,D2,D3> const &arr ) {
    typename std::remove_cv<T>::type m = arr.myData[0];
    for (int i=1; i<arr.totElems(); i++) {
      if (arr.myData[i] < m) { m = arr.myData[i]; }
    }
    return m;
  }
  template <class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE T minval( SArray<T,rank,D0,D1,D2,D3> const &arr ) {
    typename std::remove_cv<T>::type m = arr.myData[0];
    for (int i=1; i<arr.totElems(); i++) {
      if (arr.myData[i] < m) { m = arr.myData[i]; }
    }
    return m;
  }



  ////////////////////////////////////////////////////////////////////////
  // minloc (only for rank-1 stack Arrays)
  ////////////////////////////////////////////////////////////////////////
  template <class T, class D0> YAKL_INLINE int minloc( FSArray<T,1,D0> const &arr ) {
    T m = arr.myData[0];
    int loc = lbound(arr,1);
    for (int i=lbound(arr,1); i<=ubound(arr,1); i++) {
      if (arr(i) < m) {
        m = arr(i);
        loc = i;
      }
    }
    return loc;
  }
  template <class T, unsigned D0> YAKL_INLINE int minloc( SArray<T,1,D0> const &arr ) {
    T m = arr.myData[0];
    int loc = 0;
    for (int i=1; i < arr.get_dimensions()(0); i++) {
      if (arr(i) < m) {
        m = arr(i);
        loc = i;
      }
    }
    return loc;
  }



  ////////////////////////////////////////////////////////////////////////
  // maxval
  ////////////////////////////////////////////////////////////////////////
  template <class T, int rank, int myStyle>
  inline T maxval( Array<T,rank,memHost,myStyle> const &arr ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling maxval on an array that has not been initialized"); }
    #endif
    typename std::remove_cv<T>::type m = arr.myData[0];
    for (int i=1; i<arr.totElems(); i++) {
      if (arr.myData[i] > m) { m = arr.myData[i]; }
    }
    return m;
  }
  template <class T, int rank, int myStyle>
  inline T maxval( Array<T,rank,memDevice,myStyle> const &arr ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling maxval on an array that has not been initialized"); }
    #endif
    typedef typename std::remove_cv<T>::type TNC; // T Non-Const
    ParallelMax<TNC,memDevice> pmax(arr.totElems());
    return pmax( const_cast<TNC *>(arr.data()) );
  }
  template <class T, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE T maxval( FSArray<T,rank,D0,D1,D2,D3> const &arr ) {
    typename std::remove_cv<T>::type m = arr.myData[0];
    for (int i=1; i<arr.totElems(); i++) {
      if (arr.myData[i] > m) { m = arr.myData[i]; }
    }
    return m;
  }
  template <class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE T maxval( SArray<T,rank,D0,D1,D2,D3> const &arr ) {
    typename std::remove_cv<T>::type m = arr.myData[0];
    for (int i=1; i<arr.totElems(); i++) {
      if (arr.myData[i] > m) { m = arr.myData[i]; }
    }
    return m;
  }



  ////////////////////////////////////////////////////////////////////////
  // maxloc (only for rank-1 stack Arrays)
  ////////////////////////////////////////////////////////////////////////
  template <class T, class D0> YAKL_INLINE int maxloc( FSArray<T,1,D0> const &arr ) {
    T m = arr.myData[0];
    int loc = lbound(arr,1);
    for (int i=lbound(arr,1); i<=ubound(arr,1); i++) {
      if (arr(i) > m) {
        m = arr(i);
        loc = i;
      }
    }
    return loc;
  }
  template <class T, unsigned D0> YAKL_INLINE int maxloc( SArray<T,1,D0> const &arr ) {
    T m = arr.myData[0];
    int loc = 0;
    for (int i=1; i<arr.get_dimensions()(0); i++) {
      if (arr(i) > m) {
        m = arr(i);
        loc = i;
      }
    }
    return loc;
  }



  ////////////////////////////////////////////////////////////////////////
  // sum
  ////////////////////////////////////////////////////////////////////////
  template <class T, int rank, int myStyle>
  inline T sum( Array<T,rank,memHost,myStyle> const &arr ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling sum on an array that has not been initialized"); }
    #endif
    typename std::remove_cv<T>::type m = arr.myData[0];
    for (int i=1; i<arr.totElems(); i++) { m += arr.myData[i]; }
    return m;
  }
  template <class T, int rank, int myStyle>
  inline T sum( Array<T,rank,memDevice,myStyle> const &arr ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling sum on an array that has not been initialized"); }
    #endif
    typedef typename std::remove_cv<T>::type TNC;  // T Non-Const
    ParallelSum<TNC,memDevice> psum(arr.totElems());
    return psum( const_cast<TNC *>(arr.data()) );
  }
  template <class T, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE T sum( FSArray<T,rank,D0,D1,D2,D3> const &arr ) {
    typename std::remove_cv<T>::type m = arr.myData[0];
    for (int i=1; i<arr.totElems(); i++) { m += arr.myData[i]; }
    return m;
  }
  template <class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE T sum( SArray<T,rank,D0,D1,D2,D3> const &arr ) {
    typename std::remove_cv<T>::type m = arr.myData[0];
    for (int i=1; i<arr.totElems(); i++) { m += arr.myData[i]; }
    return m;
  }




  ////////////////////////////////////////////////////////////////////////
  // product (only for stack arrays)
  ////////////////////////////////////////////////////////////////////////
  template <class T, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE T product( FSArray<T,rank,D0,D1,D2,D3> const &arr ) {
    T m = arr.myData[0];
    for (int i=1; i<arr.totElems(); i++) { m *= arr.myData[i]; }
    return m;
  }
  template <class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE T product( SArray<T,rank,D0,D1,D2,D3> const &arr ) {
    T m = arr.myData[0];
    for (int i=1; i<arr.totElems(); i++) { m *= arr.myData[i]; }
    return m;
  }



  ///////////////////////////
  // any* device Array
  ///////////////////////////
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyLT ( Array<T,rank,memDevice,myStyle> const &arr , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    ScalarLiveOut<bool> ret(false);
    c::parallel_for( c::SimpleBounds<1>(arr.totElems()) , YAKL_LAMBDA (int i) {
      if ( arr.myData[i] < val ) { ret = true; }
    });
    return ret.hostRead();
  }
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyLTE ( Array<T,rank,memDevice,myStyle> const &arr , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    ScalarLiveOut<bool> ret(false);
    c::parallel_for( c::SimpleBounds<1>(arr.totElems()) , YAKL_LAMBDA (int i) {
      if ( arr.myData[i] <= val ) { ret = true; }
    });
    return ret.hostRead();
  }
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyGT ( Array<T,rank,memDevice,myStyle> const &arr , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    ScalarLiveOut<bool> ret(false);
    c::parallel_for( c::SimpleBounds<1>(arr.totElems()) , YAKL_LAMBDA (int i) {
      if ( arr.myData[i] > val ) { ret = true; }
    });
    return ret.hostRead();
  }
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyGTE ( Array<T,rank,memDevice,myStyle> const &arr , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    ScalarLiveOut<bool> ret(false);
    c::parallel_for( c::SimpleBounds<1>(arr.totElems()) , YAKL_LAMBDA (int i) {
      if ( arr.myData[i] >= val ) { ret = true; }
    });
    return ret.hostRead();
  }
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyEQ ( Array<T,rank,memDevice,myStyle> const &arr , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    ScalarLiveOut<bool> ret(false);
    c::parallel_for( c::SimpleBounds<1>(arr.totElems()) , YAKL_LAMBDA (int i) {
      if ( arr.myData[i] == val ) { ret = true; }
    });
    return ret.hostRead();
  }
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyNEQ ( Array<T,rank,memDevice,myStyle> const &arr , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    ScalarLiveOut<bool> ret(false);
    c::parallel_for( c::SimpleBounds<1>(arr.totElems()) , YAKL_LAMBDA (int i) {
      if ( arr.myData[i] != val ) { ret = true; }
    });
    return ret.hostRead();
  }


  ///////////////////////////
  // any* device Array Masked
  ///////////////////////////
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyLT ( Array<T,rank,memDevice,myStyle> const &arr , Array<bool,rank,memDevice,myStyle> const &mask , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    ScalarLiveOut<bool> ret(false);
    c::parallel_for( c::SimpleBounds<1>(arr.totElems()) , YAKL_LAMBDA (int i) {
      if ( mask.myData[i] && arr.myData[i] < val ) { ret = true; }
    });
    return ret.hostRead();
  }
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyLTE ( Array<T,rank,memDevice,myStyle> const &arr , Array<bool,rank,memDevice,myStyle> const &mask , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    ScalarLiveOut<bool> ret(false);
    c::parallel_for( c::SimpleBounds<1>(arr.totElems()) , YAKL_LAMBDA (int i) {
      if ( mask.myData[i] && arr.myData[i] <= val ) { ret = true; }
    });
    return ret.hostRead();
  }
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyGT ( Array<T,rank,memDevice,myStyle> const &arr , Array<bool,rank,memDevice,myStyle> const &mask , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    ScalarLiveOut<bool> ret(false);
    c::parallel_for( c::SimpleBounds<1>(arr.totElems()) , YAKL_LAMBDA (int i) {
      if ( mask.myData[i] && arr.myData[i] > val ) { ret = true; }
    });
    return ret.hostRead();
  }
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyGTE ( Array<T,rank,memDevice,myStyle> const &arr , Array<bool,rank,memDevice,myStyle> const &mask , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    ScalarLiveOut<bool> ret(false);
    c::parallel_for( c::SimpleBounds<1>(arr.totElems()) , YAKL_LAMBDA (int i) {
      if ( mask.myData[i] && arr.myData[i] >= val ) { ret = true; }
    });
    return ret.hostRead();
  }
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyEQ ( Array<T,rank,memDevice,myStyle> const &arr , Array<bool,rank,memDevice,myStyle> const &mask , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    ScalarLiveOut<bool> ret(false);
    c::parallel_for( c::SimpleBounds<1>(arr.totElems()) , YAKL_LAMBDA (int i) {
      if ( mask.myData[i] && arr.myData[i] == val ) { ret = true; }
    });
    return ret.hostRead();
  }
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyNEQ ( Array<T,rank,memDevice,myStyle> const &arr , Array<bool,rank,memDevice,myStyle> const &mask , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    ScalarLiveOut<bool> ret(false);
    c::parallel_for( c::SimpleBounds<1>(arr.totElems()) , YAKL_LAMBDA (int i) {
      if ( mask.myData[i] && arr.myData[i] != val ) { ret = true; }
    });
    return ret.hostRead();
  }



  ///////////////////////////
  // any* host Array
  ///////////////////////////
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyLT( Array<T,rank,memHost,myStyle> const &arr , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    bool ret = false;
    for (int i=0; i < arr.totElems(); i++) {
      if ( arr.myData[i] < val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyLTE( Array<T,rank,memHost,myStyle> const &arr , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    bool ret = false;
    for (int i=0; i < arr.totElems(); i++) {
      if ( arr.myData[i] <= val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyGT( Array<T,rank,memHost,myStyle> const &arr , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    bool ret = false;
    for (int i=0; i < arr.totElems(); i++) {
      if ( arr.myData[i] > val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyGTE( Array<T,rank,memHost,myStyle> const &arr , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    bool ret = false;
    for (int i=0; i < arr.totElems(); i++) {
      if ( arr.myData[i] >= val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyEQ( Array<T,rank,memHost,myStyle> const &arr , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    bool ret = false;
    for (int i=0; i < arr.totElems(); i++) {
      if ( arr.myData[i] == val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyNEQ( Array<T,rank,memHost,myStyle> const &arr , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    bool ret = false;
    for (int i=0; i < arr.totElems(); i++) {
      if ( arr.myData[i] != val ) { ret = true; }
    }
    return ret;
  }



  ///////////////////////////
  // any* host Array Masked
  ///////////////////////////
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyLT( Array<T,rank,memHost,myStyle> const &arr , Array<bool,rank,memHost,myStyle> const &mask , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    bool ret = false;
    for (int i=0; i < arr.totElems(); i++) {
      if ( mask.myData[i] && arr.myData[i] < val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyLTE( Array<T,rank,memHost,myStyle> const &arr , Array<bool,rank,memHost,myStyle> const &mask , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    bool ret = false;
    for (int i=0; i < arr.totElems(); i++) {
      if ( mask.myData[i] && arr.myData[i] <= val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyGT( Array<T,rank,memHost,myStyle> const &arr , Array<bool,rank,memHost,myStyle> const &mask , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    bool ret = false;
    for (int i=0; i < arr.totElems(); i++) {
      if ( mask.myData[i] && arr.myData[i] > val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyGTE( Array<T,rank,memHost,myStyle> const &arr , Array<bool,rank,memHost,myStyle> const &mask , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    bool ret = false;
    for (int i=0; i < arr.totElems(); i++) {
      if ( mask.myData[i] && arr.myData[i] >= val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyEQ( Array<T,rank,memHost,myStyle> const &arr , Array<bool,rank,memHost,myStyle> const &mask , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    bool ret = false;
    for (int i=0; i < arr.totElems(); i++) {
      if ( mask.myData[i] && arr.myData[i] == val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, int myStyle>
  inline bool anyNEQ( Array<T,rank,memHost,myStyle> const &arr , Array<bool,rank,memHost,myStyle> const &mask , TVAL val ) {
    #ifdef YAKL_DEBUG
      if (!arr.initialized()) { yakl_throw("ERROR: calling any on an array that has not been initialized"); }
    #endif
    bool ret = false;
    for (int i=0; i < arr.totElems(); i++) {
      if ( mask.myData[i] && arr.myData[i] != val ) { ret = true; }
    }
    return ret;
  }



  //////////////////////////
  // any* FSArray
  //////////////////////////
  template <class T, class TVAL, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE bool anyLT( FSArray<T,rank,D0,D1,D2,D3> const &arr , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( arr.myData[i] < val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE bool anyLTE( FSArray<T,rank,D0,D1,D2,D3> const &arr , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( arr.myData[i] <= val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE bool anyGT( FSArray<T,rank,D0,D1,D2,D3> const &arr , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( arr.myData[i] > val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE bool anyGTE( FSArray<T,rank,D0,D1,D2,D3> const &arr , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( arr.myData[i] >= val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE bool anyEQ( FSArray<T,rank,D0,D1,D2,D3> const &arr , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( arr.myData[i] == val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE bool anyNEQ( FSArray<T,rank,D0,D1,D2,D3> const &arr , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( arr.myData[i] != val ) { ret = true; }
    }
    return ret;
  }


  //////////////////////////
  // any* FSArray Masked
  //////////////////////////
  template <class T, class TVAL, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE bool anyLT( FSArray<T,rank,D0,D1,D2,D3> const &arr , FSArray<bool,rank,D0,D1,D2,D3> const &mask , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( mask.myData[i] && arr.myData[i] < val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE bool anyLTE( FSArray<T,rank,D0,D1,D2,D3> const &arr , FSArray<bool,rank,D0,D1,D2,D3> const &mask , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( mask.myData[i] && arr.myData[i] <= val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE bool anyGT( FSArray<T,rank,D0,D1,D2,D3> const &arr , FSArray<bool,rank,D0,D1,D2,D3> const &mask , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( mask.myData[i] && arr.myData[i] > val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE bool anyGTE( FSArray<T,rank,D0,D1,D2,D3> const &arr , FSArray<bool,rank,D0,D1,D2,D3> const &mask , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( mask.myData[i] && arr.myData[i] >= val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE bool anyEQ( FSArray<T,rank,D0,D1,D2,D3> const &arr , FSArray<bool,rank,D0,D1,D2,D3> const &mask , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( mask.myData[i] && arr.myData[i] == val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE bool anyNEQ( FSArray<T,rank,D0,D1,D2,D3> const &arr , FSArray<bool,rank,D0,D1,D2,D3> const &mask , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( mask.myData[i] && arr.myData[i] != val ) { ret = true; }
    }
    return ret;
  }


  //////////////////////////////
  // any* SArray
  //////////////////////////////
  template <class T, class TVAL, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE bool anyLT( SArray<T,rank,D0,D1,D2,D3> const &arr , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( arr.myData[i] < val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE bool anyLTE( SArray<T,rank,D0,D1,D2,D3> const &arr , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( arr.myData[i] <= val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE bool anyGT( SArray<T,rank,D0,D1,D2,D3> const &arr , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( arr.myData[i] > val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE bool anyGTE( SArray<T,rank,D0,D1,D2,D3> const &arr , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( arr.myData[i] >= val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE bool anyEQ( SArray<T,rank,D0,D1,D2,D3> const &arr , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( arr.myData[i] == val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE bool anyNEQ( SArray<T,rank,D0,D1,D2,D3> const &arr , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( arr.myData[i] != val ) { ret = true; }
    }
    return ret;
  }


  //////////////////////////////
  // any* SArray Masked
  //////////////////////////////
  template <class T, class TVAL, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE bool anyLT( SArray<T,rank,D0,D1,D2,D3> const &arr , SArray<bool,rank,D0,D1,D2,D3> const &mask , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( mask.myData[i] && arr.myData[i] < val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE bool anyLTE( SArray<T,rank,D0,D1,D2,D3> const &arr , SArray<bool,rank,D0,D1,D2,D3> const &mask , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( mask.myData[i] && arr.myData[i] <= val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE bool anyGT( SArray<T,rank,D0,D1,D2,D3> const &arr , SArray<bool,rank,D0,D1,D2,D3> const &mask , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( mask.myData[i] && arr.myData[i] > val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE bool anyGTE( SArray<T,rank,D0,D1,D2,D3> const &arr , SArray<bool,rank,D0,D1,D2,D3> const &mask , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( mask.myData[i] && arr.myData[i] >= val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE bool anyEQ( SArray<T,rank,D0,D1,D2,D3> const &arr , SArray<bool,rank,D0,D1,D2,D3> const &mask , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( mask.myData[i] && arr.myData[i] == val ) { ret = true; }
    }
    return ret;
  }
  template <class T, class TVAL, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE bool anyNEQ( SArray<T,rank,D0,D1,D2,D3> const &arr , SArray<bool,rank,D0,D1,D2,D3> const &mask , TVAL val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( mask.myData[i] && arr.myData[i] != val ) { ret = true; }
    }
    return ret;
  }



  ///////////////////////////////////////////////////////////
  // Matrix multiplication routines for column-row format
  ///////////////////////////////////////////////////////////
  template <class T, index_t COL_L, index_t ROW_L, index_t COL_R>
  YAKL_INLINE SArray<T,2,COL_R,ROW_L>
  matmul_cr ( SArray<T,2,COL_L,ROW_L> const &left ,
              SArray<T,2,COL_R,COL_L> const &right ) {
    SArray<T,2,COL_R,ROW_L> ret;
    for (index_t i=0; i < COL_R; i++) {
      for (index_t j=0; j < ROW_L; j++) {
        T tmp = 0;
        for (index_t k=0; k < COL_L; k++) {
          tmp += left(k,j) * right(i,k);
        }
        ret(i,j) = tmp;
      }
    }
    return ret;
  }


  template<class T, index_t COL_L, index_t ROW_L>
  YAKL_INLINE SArray<T,1,ROW_L>
  matmul_cr ( SArray<T,2,COL_L,ROW_L> const &left ,
              SArray<T,1,COL_L>       const &right ) {
    SArray<T,1,ROW_L> ret;
    for (index_t j=0; j < ROW_L; j++) {
      T tmp = 0;
      for (index_t k=0; k < COL_L; k++) {
        tmp += left(k,j) * right(k);
      }
      ret(j) = tmp;
    }
    return ret;
  }


  template <class T, int COL_L, int ROW_L, int COL_R>
  YAKL_INLINE FSArray<T,2,SB<COL_R>,SB<ROW_L>>
  matmul_cr ( FSArray<T,2,SB<COL_L>,SB<ROW_L>> const &left ,
              FSArray<T,2,SB<COL_R>,SB<COL_L>> const &right ) {
    FSArray<T,2,SB<COL_R>,SB<ROW_L>> ret;
    for (index_t i=1; i <= COL_R; i++) {
      for (index_t j=1; j <= ROW_L; j++) {
        T tmp = 0;
        for (index_t k=1; k <= COL_L; k++) {
          tmp += left(k,j) * right(i,k);
        }
        ret(i,j) = tmp;
      }
    }
    return ret;
  }


  template<class T, int COL_L, int ROW_L>
  YAKL_INLINE FSArray<T,1,SB<ROW_L>>
  matmul_cr ( FSArray<T,2,SB<COL_L>,SB<ROW_L>> const &left ,
              FSArray<T,1,SB<COL_L>>           const &right ) {
    FSArray<T,1,SB<ROW_L>> ret;
    for (index_t j=1; j <= ROW_L; j++) {
      T tmp = 0;
      for (index_t k=1; k <= COL_L; k++) {
        tmp += left(k,j) * right(k);
      }
      ret(j) = tmp;
    }
    return ret;
  }


  ///////////////////////////////////////////////////////////
  // Matrix multiplication routines for row-column format
  ///////////////////////////////////////////////////////////
  template <class T, index_t COL_L, index_t ROW_L, index_t COL_R>
  YAKL_INLINE SArray<T,2,ROW_L,COL_R>
  matmul_rc ( SArray<T,2,ROW_L,COL_L> const &left ,
              SArray<T,2,COL_L,COL_R> const &right ) {
    SArray<T,2,ROW_L,COL_R> ret;
    for (index_t i=0; i < COL_R; i++) {
      for (index_t j=0; j < ROW_L; j++) {
        T tmp = 0;
        for (index_t k=0; k < COL_L; k++) {
          tmp += left(j,k) * right(k,i);
        }
        ret(j,i) = tmp;
      }
    }
    return ret;
  }


  template<class T, index_t COL_L, index_t ROW_L>
  YAKL_INLINE SArray<T,1,ROW_L>
  matmul_rc ( SArray<T,2,ROW_L,COL_L> const &left ,
              SArray<T,1,COL_L>       const &right ) {
    SArray<T,1,ROW_L> ret;
    for (index_t j=0; j < ROW_L; j++) {
      T tmp = 0;
      for (index_t k=0; k < COL_L; k++) {
        tmp += left(j,k) * right(k);
      }
      ret(j) = tmp;
    }
    return ret;
  }


  template <class T, int COL_L, int ROW_L, int COL_R>
  YAKL_INLINE FSArray<T,2,SB<ROW_L>,SB<COL_R>>
  matmul_rc ( FSArray<T,2,SB<ROW_L>,SB<COL_L>> const &left ,
              FSArray<T,2,SB<COL_L>,SB<COL_R>> const &right ) {
    FSArray<T,2,SB<ROW_L>,SB<COL_R>> ret;
    for (index_t i=1; i <= COL_R; i++) {
      for (index_t j=1; j <= ROW_L; j++) {
        T tmp = 0;
        for (index_t k=1; k <= COL_L; k++) {
          tmp += left(j,k) * right(k,i);
        }
        ret(j,i) = tmp;
      }
    }
    return ret;
  }


  template<class T, int COL_L, int ROW_L>
  YAKL_INLINE FSArray<T,1,SB<ROW_L>>
  matmul_rc ( FSArray<T,2,SB<ROW_L>,SB<COL_L>> const &left ,
              FSArray<T,1,SB<COL_L>>           const &right ) {
    FSArray<T,1,SB<ROW_L>> ret;
    for (index_t j=1; j <= ROW_L; j++) {
      T tmp = 0;
      for (index_t k=1; k <= COL_L; k++) {
        tmp += left(j,k) * right(k);
      }
      ret(j) = tmp;
    }
    return ret;
  }



  /////////////////////////////////////////////////////////////////
  // Matrix inverse with Gaussian Elimination (no pivoting)
  // for row-column format
  /////////////////////////////////////////////////////////////////
  template <unsigned int n, class real>
  YAKL_INLINE SArray<real,2,n,n> matinv_ge(SArray<real,2,n,n> const &a) {
    SArray<real,2,n,n> scratch;
    SArray<real,2,n,n> inv;

    // Initialize inverse as identity
    for (int icol = 0; icol < n; icol++) {
      for (int irow = 0; irow < n; irow++) {
        scratch(icol,irow) = a(icol,irow);
        if (icol == irow) {
          inv(irow,icol) = 1;
        } else {
          inv(irow,icol) = 0;
        }
      }
    }

    // Gaussian elimination to zero out lower
    for (int idiag = 0; idiag < n; idiag++) {
      // Divide out the diagonal component from the first row
      real factor = static_cast<real>(1)/scratch(idiag,idiag);
      for (int icol = idiag; icol < n; icol++) {
        scratch(idiag,icol) *= factor;
      }
      for (int icol = 0; icol < n; icol++) {
        inv(idiag,icol) *= factor;
      }
      for (int irow = idiag+1; irow < n; irow++) {
        real factor = scratch(irow,idiag);
        for (int icol = idiag; icol < n; icol++) {
          scratch(irow,icol) -= factor * scratch(idiag,icol);
        }
        for (int icol = 0; icol < n; icol++) {
          inv    (irow,icol) -= factor * inv    (idiag,icol);
        }
      }
    }

    // Gaussian elimination to zero out upper
    for (int idiag = n-1; idiag >= 1; idiag--) {
      for (int irow = 0; irow < idiag; irow++) {
        real factor = scratch(irow,idiag);
        for (int icol = irow+1; icol < n; icol++) {
          scratch(irow,icol) -= factor * scratch(idiag,icol);
        }
        for (int icol = 0; icol < n; icol++) {
          inv    (irow,icol) -= factor * inv    (idiag,icol);
        }
      }
    }

    return inv;
  }

  template <int n, class real>
  YAKL_INLINE FSArray<real,2,SB<n>,SB<n>> matinv_ge(FSArray<real,2,SB<n>,SB<n>> const &a) {
    FSArray<real,2,SB<n>,SB<n>> scratch;
    FSArray<real,2,SB<n>,SB<n>> inv;

    // Initialize inverse as identity
    for (int icol = 0; icol < n; icol++) {
      for (int irow = 0; irow < n; irow++) {
        scratch(icol+1,irow+1) = a(icol+1,irow+1);
        if (icol == irow) {
          inv(irow+1,icol+1) = 1;
        } else {
          inv(irow+1,icol+1) = 0;
        }
      }
    }

    // Gaussian elimination to zero out lower
    for (int idiag = 0; idiag < n; idiag++) {
      // Divide out the diagonal component from the first row
      real factor = static_cast<real>(1)/scratch(idiag+1,idiag+1);
      for (int icol = idiag; icol < n; icol++) {
        scratch(idiag+1,icol+1) *= factor;
      }
      for (int icol = 0; icol < n; icol++) {
        inv(idiag+1,icol+1) *= factor;
      }
      for (int irow = idiag+1; irow < n; irow++) {
        real factor = scratch(irow+1,idiag+1);
        for (int icol = idiag; icol < n; icol++) {
          scratch(irow+1,icol+1) -= factor * scratch(idiag+1,icol+1);
        }
        for (int icol = 0; icol < n; icol++) {
          inv    (irow+1,icol+1) -= factor * inv    (idiag+1,icol+1);
        }
      }
    }

    // Gaussian elimination to zero out upper
    for (int idiag = n-1; idiag >= 1; idiag--) {
      for (int irow = 0; irow < idiag; irow++) {
        real factor = scratch(irow+1,idiag+1);
        for (int icol = irow+1; icol < n; icol++) {
          scratch(irow+1,icol+1) -= factor * scratch(idiag+1,icol+1);
        }
        for (int icol = 0; icol < n; icol++) {
          inv    (irow+1,icol+1) -= factor * inv    (idiag+1,icol+1);
        }
      }
    }

    return inv;
  }


  /////////////////////////////////////////////////////////////////
  // Transpose
  /////////////////////////////////////////////////////////////////
  template <unsigned int n1, unsigned int n2, class real>
  YAKL_INLINE SArray<real,2,n2,n1> transpose(SArray<real,2,n1,n2> const &a) {
    SArray<real,2,n2,n1> ret;
    for (int j=0; j < n1; j++) {
      for (int i=0; i < n2; i++) {
        ret(j,i) = a(i,j);
      }
    }
    return ret;
  }
  template <int n1, int n2, class real>
  YAKL_INLINE FSArray<real,2,SB<n2>,SB<n1>> transpose(FSArray<real,2,SB<n1>,SB<n2>> const &a) {
    FSArray<real,2,SB<n2>,SB<n1>> ret;
    for (int j=1; j <= n1; j++) {
      for (int i=1; i <= n2; i++) {
        ret(j,i) = a(i,j);
      }
    }
    return ret;
  }



  /////////////////////////////////////////////////////////////////
  // Count
  /////////////////////////////////////////////////////////////////
  template <int rank, int myStyle>
  inline int count( Array<bool,rank,memHost,myStyle> const &mask ) {
    #ifdef YAKL_DEBUG
      if (!mask.initialized()) { yakl_throw("ERROR: calling count on an array that has not been initialized"); }
    #endif
    int numTrue = 0;
    for (int i=0; i < mask.totElems(); i++) {
      if (mask.myData[i]) { numTrue++; }
    }
    return numTrue;
  }
  template <int rank, int myStyle>
  inline int count( Array<bool,rank,memDevice,myStyle> const &mask ) {
    #ifdef YAKL_DEBUG
      if (!mask.initialized()) { yakl_throw("ERROR: calling count on an array that has not been initialized"); }
    #endif
    ScalarLiveOut<int> numTrue(0);
    c::parallel_for( c::SimpleBounds<1>( mask.totElems() ) , YAKL_DEVICE_LAMBDA (int i) {
      if (mask.myData[i]) { atomicAdd(numTrue(),1); }
    });
    return numTrue.hostRead();
  }
  template <int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE int count( FSArray<bool,rank,D0,D1,D2,D3> const &mask ) {
    int numTrue = 0;
    for (int i=0; i < mask.totElems(); i++) {
      if (mask.myData[i]) { numTrue++; }
    }
    return numTrue;
  }
  template <int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE int count( SArray<bool,rank,D0,D1,D2,D3> const &mask ) {
    int numTrue = 0;
    for (int i=0; i < mask.totElems(); i++) {
      if (mask.myData[i]) { numTrue++; }
    }
    return numTrue;
  }



  /////////////////////////////////////////////////////////////////
  // Pack
  /////////////////////////////////////////////////////////////////
  template <class T, int rank, int myStyle>
  inline Array<T,1,memHost,myStyle> pack( Array<T,rank,memHost,myStyle> const &arr ,
                                          Array<bool,rank,memHost,myStyle> const &mask = Array<bool,rank,memHost,myStyle>() ) {
    if (allocated(mask)) {
      if (mask.totElems() != arr.totElems()) {
        yakl_throw("Error: pack: arr and mask have a different number of elements");
      }
      // count the number of true elements
      int numTrue = count( mask );
      Array<T,1,memHost,myStyle> ret("packReturn",numTrue);
      int slot = 0;
      for (int i=0; i < arr.totElems(); i++) {
        if (mask.myData[i]) { ret.myData[slot] = arr.myData[i]; slot++; }
      }
      return ret;
    } else {
      Array<T,1,memHost,myStyle> ret("packReturn",arr.totElems());
      for (int i=0; i < arr.totElems(); i++) {
        ret.myData[i] = arr.myData[i];
      }
      return ret;
    }
  }



}


