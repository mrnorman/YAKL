
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


  template <class T, index_t COL_L, index_t ROW_L, index_t COL_R>
  YAKL_INLINE FSArray<T,2,SB<COL_R>,SB<ROW_L>>
  matmul_cr ( FSArray<T,2,SB<COL_L>,SB<ROW_L>> const &left ,
              FSArray<T,2,SB<COL_R>,SB<COL_L>> const &right ) {
    FSArray<T,2,SB<COL_R>,SB<ROW_L>> ret;
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
  YAKL_INLINE FSArray<T,1,SB<ROW_L>>
  matmul_cr ( FSArray<T,2,SB<COL_L>,SB<ROW_L>> const &left ,
              FSArray<T,1,SB<COL_L>>           const &right ) {
    FSArray<T,1,SB<ROW_L>> ret;
    for (index_t j=0; j < ROW_L; j++) {
      T tmp = 0;
      for (index_t k=0; k < COL_L; k++) {
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


  template <class T, index_t COL_L, index_t ROW_L, index_t COL_R>
  YAKL_INLINE FSArray<T,2,SB<ROW_L>,SB<COL_R>>
  matmul_rc ( FSArray<T,2,SB<ROW_L>,SB<COL_L>> const &left ,
              FSArray<T,2,SB<COL_L>,SB<COL_R>> const &right ) {
    FSArray<T,2,SB<ROW_L>,SB<COL_R>> ret;
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
  YAKL_INLINE FSArray<T,1,SB<ROW_L>>
  matmul_rc ( FSArray<T,2,SB<ROW_L>,SB<COL_L>> const &left ,
              FSArray<T,1,SB<COL_L>>           const &right ) {
    FSArray<T,1,SB<ROW_L>> ret;
    for (index_t j=0; j < ROW_L; j++) {
      T tmp = 0;
      for (index_t k=0; k < COL_L; k++) {
        tmp += left(j,k) * right(k);
      }
      ret(j) = tmp;
    }
    return ret;
  }




  /////////////////////////////////////////////////////////////////
  // Matrix inverse with Gaussian Elimination (no pivoting)
  // for column-row format
  /////////////////////////////////////////////////////////////////
  template <unsigned int n, class real>
  YAKL_INLINE SArray<real,2,n,n> matinv_ge_cr(SArray<real,2,n,n> const &a) {
    SArray<real,2,n,n> scratch;
    SArray<real,2,n,n> inv;

    // Initialize inverse as identity
    for (int icol = 0; icol < n; icol++) {
      for (int irow = 0; irow < n; irow++) {
        scratch(icol,irow) = a(icol,irow);
        if (icol == irow) {
          inv(icol,irow) = 1;
        } else {
          inv(icol,irow) = 0;
        }
      }
    }

    // Gaussian elimination to zero out lower
    for (int idiag = 0; idiag < n; idiag++) {
      // Divide out the diagonal component from the first row
      real factor = static_cast<real>(1)/scratch(idiag,idiag);
      for (int icol = idiag; icol < n; icol++) {
        scratch(icol,idiag) *= factor;
      }
      for (int icol = 0; icol < n; icol++) {
        inv(icol,idiag) *= factor;
      }
      for (int irow = idiag+1; irow < n; irow++) {
        real factor = scratch(idiag,irow);
        for (int icol = idiag; icol < n; icol++) {
          scratch(icol,irow) -= factor * scratch(icol,idiag);
        }
        for (int icol = 0; icol < n; icol++) {
          inv    (icol,irow) -= factor * inv    (icol,idiag);
        }
      }
    }

    // Gaussian elimination to zero out upper
    for (int idiag = n-1; idiag >= 1; idiag--) {
      for (int irow = 0; irow < idiag; irow++) {
        real factor = scratch(idiag,irow);
        for (int icol = irow+1; icol < n; icol++) {
          scratch(icol,irow) -= factor * scratch(icol,idiag);
        }
        for (int icol = 0; icol < n; icol++) {
          inv    (icol,irow) -= factor * inv    (icol,idiag);
        }
      }
    }

    return inv;
  }


  /////////////////////////////////////////////////////////////////
  // Matrix inverse with Gaussian Elimination (no pivoting)
  // for row-column format
  /////////////////////////////////////////////////////////////////
  template <unsigned int n, class real>
  YAKL_INLINE SArray<real,2,n,n> matinv_ge_rc(SArray<real,2,n,n> const &a) {
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


