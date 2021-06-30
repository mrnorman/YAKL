
#pragma once

namespace intrinsics {

  template <class T> YAKL_INLINE int  size(T const &arr, int dim) { return arr.get_dimensions()(dim); }
  template <class T> YAKL_INLINE int  size(T const &arr) { return arr.totElems(); }



  template <class T> YAKL_INLINE auto shape(T const &arr) -> decltype(arr.get_dimensions()) { return arr.get_dimensions(); }



  template <class T> YAKL_INLINE bool allocated (T const &arr) { return arr.myData != nullptr; }



  template <class T> YAKL_INLINE bool associated (T const &arr) { return arr.myData != nullptr; }



  template <class T> YAKL_INLINE int  lbound (T const &arr, int dim) { return arr.get_lbounds()(dim); }
  template <class T> YAKL_INLINE auto lbound (T const &arr) -> decltype(arr.get_lbounds()) { return arr.get_lbounds(); }



  template <class T> YAKL_INLINE int  ubound (T const &arr, int dim) { return arr.get_ubounds()(dim); }
  template <class T> YAKL_INLINE auto ubound (T const &arr) -> decltype(arr.get_ubounds()) { return arr.get_ubounds(); }



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
    T m = arr.myData[0];
    for (int i=1; i<arr.totElems(); i++) {
      if (arr.myData[i] < m) { m = arr.myData[i]; }
    }
    return m;
  }
  template <class T, int rank, int myStyle>
  inline T minval( Array<T,rank,memDevice,myStyle> const &arr ) {
    ParallelMin<T,memDevice> pmin(arr.totElems());
    return pmin( arr.data() );
  }
  template <class T, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE T minval( FSArray<T,rank,D0,D1,D2,D3> const &arr ) {
    T m = arr.myData[0];
    for (int i=1; i<arr.totElems(); i++) {
      if (arr.myData[i] < m) { m = arr.myData[i]; }
    }
    return m;
  }
  template <class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE T minval( SArray<T,rank,D0,D1,D2,D3> const &arr ) {
    T m = arr.myData[0];
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
    T m = arr.myData[0];
    for (int i=1; i<arr.totElems(); i++) {
      if (arr.myData[i] > m) { m = arr.myData[i]; }
    }
    return m;
  }
  template <class T, int rank, int myStyle>
  inline T maxval( Array<T,rank,memDevice,myStyle> const &arr ) {
    ParallelMax<T,memDevice> pmax(arr.totElems());
    return pmax( arr.data() );
  }
  template <class T, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE T maxval( FSArray<T,rank,D0,D1,D2,D3> const &arr ) {
    T m = arr.myData[0];
    for (int i=1; i<arr.totElems(); i++) {
      if (arr.myData[i] > m) { m = arr.myData[i]; }
    }
    return m;
  }
  template <class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE T maxval( SArray<T,rank,D0,D1,D2,D3> const &arr ) {
    T m = arr.myData[0];
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
    for (int i=1; i<arr.dimension[0]; i++) {
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
    T m = arr.myData[0];
    for (int i=1; i<arr.totElems(); i++) { m += arr.myData[i]; }
    return m;
  }
  template <class T, int rank, int myStyle>
  inline T sum( Array<T,rank,memDevice,myStyle> const &arr ) {
    ParallelSum<T,memDevice> psum(arr.totElems());
    return psum( arr.data() );
  }
  template <class T, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE T sum( FSArray<T,rank,D0,D1,D2,D3> const &arr ) {
    T m = arr.myData[0];
    for (int i=1; i<arr.totElems(); i++) { m += arr.myData[i]; }
    return m;
  }
  template <class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE T sum( SArray<T,rank,D0,D1,D2,D3> const &arr ) {
    T m = arr.myData[0];
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



  template <class F, class T, int rank, int myStyle>
  inline bool any( Array<T,rank,yakl::memDevice,myStyle> const &arr , F const &f , T val ) {
    yakl::ScalarLiveOut<bool> ret(false);
    yakl::c::parallel_for( yakl::c::SimpleBounds<1>(arr.totElems()) , YAKL_LAMBDA (int i) {
      if ( f( arr.myData[i] , val ) ) { ret = true; }
    });
    return ret.hostRead();
  }
  template <class F, class T, int rank, int myStyle>
  inline bool any( Array<T,rank,yakl::memHost,myStyle> const &arr , F const &f , T val ) {
    bool ret = false;
    for (int i=0; i < arr.totElems(); i++) {
      if ( f( arr.myData[i] , val ) ) { ret = true; }
    }
    return ret;
  }
  template <class T, int rank, int myMem, int myStyle>
  YAKL_INLINE bool anyLT ( Array<T,rank,myMem,myStyle> const &arr , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem <  val; };
    return any( arr , test , val );
  }
  template <class T, int rank, int myMem, int myStyle>
  YAKL_INLINE bool anyLTE( Array<T,rank,myMem,myStyle> const &arr , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem <= val; };
    return any( arr , test , val );
  }
  template <class T, int rank, int myMem, int myStyle>
  YAKL_INLINE bool anyGT ( Array<T,rank,myMem,myStyle> const &arr , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem >  val; };
    return any( arr , test , val );
  }
  template <class T, int rank, int myMem, int myStyle>
  YAKL_INLINE bool anyGTE( Array<T,rank,myMem,myStyle> const &arr , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem >= val; };
    return any( arr , test , val );
  }
  template <class T, int rank, int myMem, int myStyle>
  YAKL_INLINE bool anyEQ ( Array<T,rank,myMem,myStyle> const &arr , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem == val; };
    return any( arr , test , val );
  }
  template <class T, int rank, int myMem, int myStyle>
  YAKL_INLINE bool anyNEQ ( Array<T,rank,myMem,myStyle> const &arr , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem != val; };
    return any( arr , test , val );
  }



  template <class F, class T, int rank, int myStyle>
  inline bool any( Array<T,rank,yakl::memDevice,myStyle> const &arr ,
                   Array<bool,rank,yakl::memDevice,myStyle> const &mask , F const &f , T val ) {
    yakl::ScalarLiveOut<bool> ret(false);
    yakl::c::parallel_for( yakl::c::SimpleBounds<1>(arr.totElems()) , YAKL_LAMBDA (int i) {
      if ( mask.myData[i] && f( arr.myData[i] , val ) ) { ret = true; }
    });
    return ret.hostRead();
  }
  template <class F, class T, int rank, int myStyle>
  YAKL_INLINE bool any( Array<T,rank,yakl::memHost,myStyle> const &arr ,
                        Array<bool,rank,yakl::memHost,myStyle> const &mask , F const &f , T val ) {
    bool ret = false;
    for (int i=0; i < arr.totElems(); i++) {
      if ( mask.myData[i] && f( arr.myData[i] , val ) ) { ret = true; }
    }
    return ret;
  }
  template <class T, int rank, int myMem, int myStyle>
  YAKL_INLINE bool anyLT ( Array<T,rank,myMem,myStyle> const &arr , Array<bool,rank,myMem,myStyle> const &mask , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem <  val; };
    return any( arr , mask , test , val );
  }
  template <class T, int rank, int myMem, int myStyle>
  YAKL_INLINE bool anyLTE( Array<T,rank,myMem,myStyle> const &arr , Array<bool,rank,myMem,myStyle> const &mask , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem <= val; };
    return any( arr , mask , test , val );
  }
  template <class T, int rank, int myMem, int myStyle>
  YAKL_INLINE bool anyGT ( Array<T,rank,myMem,myStyle> const &arr , Array<bool,rank,myMem,myStyle> const &mask , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem >  val; };
    return any( arr , mask , test , val );
  }
  template <class T, int rank, int myMem, int myStyle>
  YAKL_INLINE bool anyGTE( Array<T,rank,myMem,myStyle> const &arr , Array<bool,rank,myMem,myStyle> const &mask , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem >= val; };
    return any( arr , mask , test , val );
  }
  template <class T, int rank, int myMem, int myStyle>
  YAKL_INLINE bool anyEQ ( Array<T,rank,myMem,myStyle> const &arr , Array<bool,rank,myMem,myStyle> const &mask , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem == val; };
    return any( arr , mask , test , val );
  }
  template <class T, int rank, int myMem, int myStyle>
  YAKL_INLINE bool anyNEQ ( Array<T,rank,myMem,myStyle> const &arr , Array<bool,rank,myMem,myStyle> const &mask , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem != val; };
    return any( arr , mask , test , val );
  }



  template <class F, class T, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE bool any( FSArray<T,rank,D0,D1,D2,D3> const &arr , F const &f , T val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( f( arr.myData[i] , val ) ) { ret = true; }
    }
    return ret;
  }
  template <class T, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE bool anyLT ( FSArray<T,rank,D0,D1,D2,D3> const &arr , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem <  val; };
    return any( arr , test , val );
  }
  template <class T, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE bool anyLTE( FSArray<T,rank,D0,D1,D2,D3> const &arr , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem <= val; };
    return any( arr , test , val );
  }
  template <class T, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE bool anyGT ( FSArray<T,rank,D0,D1,D2,D3> const &arr , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem >  val; };
    return any( arr , test , val );
  }
  template <class T, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE bool anyGTE( FSArray<T,rank,D0,D1,D2,D3> const &arr , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem >= val; };
    return any( arr , test , val );
  }
  template <class T, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE bool anyEQ ( FSArray<T,rank,D0,D1,D2,D3> const &arr , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem == val; };
    return any( arr , test , val );
  }
  template <class T, int rank, class D0, class D1, class D2, class D3>
  YAKL_INLINE bool anyNEQ ( FSArray<T,rank,D0,D1,D2,D3> const &arr , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem != val; };
    return any( arr , test , val );
  }



  template <class F, class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE bool any( SArray<T,rank,D0,D1,D2,D3> const &arr , F const &f , T val ) {
    bool ret = false;
    for (int i=0; i<arr.totElems(); i++) {
      if ( f( arr.myData[i] , val ) ) { ret = true; }
    }
    return ret;
  }
  template <class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE bool anyLT ( SArray<T,rank,D0,D1,D2,D3> const &arr , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem <  val; };
    return any( arr , test , val );
  }
  template <class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE bool anyLTE( SArray<T,rank,D0,D1,D2,D3> const &arr , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem <= val; };
    return any( arr , test , val );
  }
  template <class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE bool anyGT ( SArray<T,rank,D0,D1,D2,D3> const &arr , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem >  val; };
    return any( arr , test , val );
  }
  template <class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE bool anyGTE( SArray<T,rank,D0,D1,D2,D3> const &arr , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem >= val; };
    return any( arr , test , val );
  }
  template <class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE bool anyEQ ( SArray<T,rank,D0,D1,D2,D3> const &arr , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem == val; };
    return any( arr , test , val );
  }
  template <class T, int rank, unsigned D0, unsigned D1, unsigned D2, unsigned D3>
  YAKL_INLINE bool anyNEQ ( SArray<T,rank,D0,D1,D2,D3> const &arr , T val ) {
    auto test = YAKL_LAMBDA (T elem , T val)->bool { return elem != val; };
    return any( arr , test , val );
  }



  template <class T, int D0_L, int D1_L, int D1_R>
  YAKL_INLINE FSArray<T,2,SB<D0_L>,SB<D1_R>>
  matmul( FSArray<T,2,SB<D0_L>,SB<D1_L>> const &a1 ,
          FSArray<T,2,SB<D1_L>,SB<D1_R>> const &a2 ) {
    FSArray<T,2,SB<D0_L>,SB<D1_R>> ret;
    for (int i=1; i <= D0_L; i++) {
      for (int j=1; j <= D1_R; j++) {
        T tmp = 0;
        for (int k=1; k <= D1_L; k++) {
          tmp += a1(i,k) * a2(k,j);
        }
        ret(i,j) = tmp;
      }
    }
    return ret;
  }


  template <class T, int D0_L, int D1_L>
  YAKL_INLINE FSArray<T,1,SB<D0_L>>
  matmul( FSArray<T,2,SB<D0_L>,SB<D1_L>> const &a1 ,
          FSArray<T,1,SB<D1_L>> const &a2 ) {
    FSArray<T,1,SB<D0_L>> ret;
    for (int i=1; i <= D0_L; i++) {
      T tmp = 0;
      for (int k=1; k <= D1_L; k++) {
        tmp += a1(i,k) * a2(k);
      }
      ret(i) = tmp;
    }
    return ret;
  }



  template <int rank, int myStyle>
  inline int count( Array<bool,rank,memHost,myStyle> const &mask ) {
    int numTrue = 0;
    for (int i=0; i < mask.totElems(); i++) {
      if (mask.myData[i]) { numTrue++; }
    }
    return numTrue;
  }
  template <int rank, int myStyle>
  inline int count( Array<bool,rank,memDevice,myStyle> const &mask ) {
    yakl::ScalarLiveOut<int> numTrue(0);
    yakl::c::parallel_for( yakl::c::SimpleBounds<1>( mask.totElems() ) , YAKL_LAMBDA (int i) {
      if (mask.myData[i]) { yakl::atomicAdd(numTrue(),1); }
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



  template <class T, int rank, int myStyle> inline Array<T,1,memHost,myStyle> pack( Array<T,rank,memHost,myStyle> const &arr ,
                                                                                    Array<bool,rank,memHost,myStyle> const &mask = 
                                                                                                Array<bool,rank,memHost,myStyle>() ) {
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


