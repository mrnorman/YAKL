
#pragma once
// Included by YAKL_Array.h

namespace yakl {

  // [S]tatic (compile-time) Array [B]ounds (templated)
  // It's only used for Fortran, so it takes on Fortran defaults
  // with lower bound default to 1
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

  template <class T, int rank, class B0, class B1=SB<1>, class B2=SB<1>, class B3=SB<1>>
  class FSArray {
  public :
    static int constexpr U0 = B0::upper();
    static int constexpr L0 = B0::lower();
    static int constexpr U1 = B1::upper();
    static int constexpr L1 = B1::lower();
    static int constexpr U2 = B2::upper();
    static int constexpr L2 = B2::lower();
    static int constexpr U3 = B3::upper();
    static int constexpr L3 = B3::lower();

    static unsigned constexpr D0 =             U0 - L0 + 1;
    static unsigned constexpr D1 = rank >= 1 ? U1 - L1 + 1 : 1;
    static unsigned constexpr D2 = rank >= 1 ? U2 - L2 + 1 : 1;
    static unsigned constexpr D3 = rank >= 1 ? U3 - L3 + 1 : 1;

    static unsigned constexpr OFF0 = 1;
    static unsigned constexpr OFF1 = D0;
    static unsigned constexpr OFF2 = D0*D1;
    static unsigned constexpr OFF3 = D0*D1*D2;

    typedef typename std::remove_cv<T>::type       type;
    typedef          T                             value_type;
    typedef typename std::add_const<type>::type    const_value_type;
    typedef typename std::remove_const<type>::type non_const_value_type;

    T mutable myData[D0*D1*D2*D3];

    // All copies are deep, so be wary of copies. Use references where possible
    YAKL_INLINE FSArray() {}
    YAKL_INLINE FSArray           (FSArray      &&in) { for (uint i=0; i < totElems(); i++) { myData[i] = in.myData[i]; } }
    YAKL_INLINE FSArray           (FSArray const &in) { for (uint i=0; i < totElems(); i++) { myData[i] = in.myData[i]; } }
    YAKL_INLINE FSArray &operator=(FSArray      &&in) { for (uint i=0; i < totElems(); i++) { myData[i] = in.myData[i]; }; return *this; }
    YAKL_INLINE FSArray &operator=(FSArray const &in) { for (uint i=0; i < totElems(); i++) { myData[i] = in.myData[i]; }; return *this; }
    YAKL_INLINE ~FSArray() { }

    YAKL_INLINE T &operator()(int const i0) const {
      static_assert(rank==1,"ERROR: Improper number of dimensions specified in operator()");
      #ifdef YAKL_DEBUG
        #if YAKL_CURRENTLY_ON_HOST()
          if constexpr (rank >= 1) { if (i0<L0 || i0>U0) { printf("FSArray i0 out of bounds (i0: %d; lb0: %d; ub0: %d",i0,L0,U0); yakl_throw(""); } }
        #else
          if constexpr (rank >= 1) { if (i0<L0 || i0>U0) { yakl_throw("ERROR: FSArray index out of bounds"); } }
        #endif
      #endif
      return myData[i0-L0];
    }
    YAKL_INLINE T &operator()(int const i0, int const i1) const {
      static_assert(rank==2,"ERROR: Improper number of dimensions specified in operator()");
      #ifdef YAKL_DEBUG
        #if YAKL_CURRENTLY_ON_HOST()
          if constexpr (rank >= 1) { if (i0<L0 || i0>U0) { printf("FSArray i0 out of bounds (i0: %d; lb0: %d; ub0: %d",i0,L0,U0); yakl_throw(""); } }
          if constexpr (rank >= 2) { if (i1<L1 || i1>U1) { printf("FSArray i1 out of bounds (i1: %d; lb1: %d; ub1: %d",i1,L1,U1); yakl_throw(""); } }
        #else
          if constexpr (rank >= 1) { if (i0<L0 || i0>U0) { yakl_throw("ERROR: FSArray index out of bounds"); } }
          if constexpr (rank >= 2) { if (i1<L1 || i1>U1) { yakl_throw("ERROR: FSArray index out of bounds"); } }
        #endif
      #endif
      return myData[(i1-L1)*OFF1 + i0-L0];
    }
    YAKL_INLINE T &operator()(int const i0, int const i1, int const i2) const {
      static_assert(rank==3,"ERROR: Improper number of dimensions specified in operator()");
      #ifdef YAKL_DEBUG
        #if YAKL_CURRENTLY_ON_HOST()
          if constexpr (rank >= 1) { if (i0<L0 || i0>U0) { printf("FSArray i0 out of bounds (i0: %d; lb0: %d; ub0: %d",i0,L0,U0); yakl_throw(""); } }
          if constexpr (rank >= 2) { if (i1<L1 || i1>U1) { printf("FSArray i1 out of bounds (i1: %d; lb1: %d; ub1: %d",i1,L1,U1); yakl_throw(""); } }
          if constexpr (rank >= 3) { if (i2<L2 || i2>U2) { printf("FSArray i2 out of bounds (i2: %d; lb2: %d; ub2: %d",i2,L2,U2); yakl_throw(""); } }
        #else
          if constexpr (rank >= 1) { if (i0<L0 || i0>U0) { yakl_throw("ERROR: FSArray index out of bounds"); } }
          if constexpr (rank >= 2) { if (i1<L1 || i1>U1) { yakl_throw("ERROR: FSArray index out of bounds"); } }
          if constexpr (rank >= 3) { if (i2<L2 || i2>U2) { yakl_throw("ERROR: FSArray index out of bounds"); } }
        #endif
      #endif
      return myData[(i2-L2)*OFF2 + (i1-L1)*OFF1 + i0-L0];
    }
    YAKL_INLINE T &operator()(int const i0, int const i1, int const i2, int const i3) const {
      static_assert(rank==4,"ERROR: Improper number of dimensions specified in operator()");
      #ifdef YAKL_DEBUG
        #if YAKL_CURRENTLY_ON_HOST()
          if constexpr (rank >= 1) { if (i0<L0 || i0>U0) { printf("FSArray i0 out of bounds (i0: %d; lb0: %d; ub0: %d",i0,L0,U0); yakl_throw(""); } }
          if constexpr (rank >= 2) { if (i1<L1 || i1>U1) { printf("FSArray i1 out of bounds (i1: %d; lb1: %d; ub1: %d",i1,L1,U1); yakl_throw(""); } }
          if constexpr (rank >= 3) { if (i2<L2 || i2>U2) { printf("FSArray i2 out of bounds (i2: %d; lb2: %d; ub2: %d",i2,L2,U2); yakl_throw(""); } }
          if constexpr (rank >= 4) { if (i3<L3 || i3>U3) { printf("FSArray i3 out of bounds (i3: %d; lb3: %d; ub3: %d",i3,L3,U3); yakl_throw(""); } }
        #else
          if constexpr (rank >= 1) { if (i0<L0 || i0>U0) { yakl_throw("ERROR: FSArray index out of bounds"); } }
          if constexpr (rank >= 2) { if (i1<L1 || i1>U1) { yakl_throw("ERROR: FSArray index out of bounds"); } }
          if constexpr (rank >= 3) { if (i2<L2 || i2>U2) { yakl_throw("ERROR: FSArray index out of bounds"); } }
          if constexpr (rank >= 4) { if (i3<L3 || i3>U3) { yakl_throw("ERROR: FSArray index out of bounds"); } }
        #endif
      #endif
      return myData[(i3-L3)*OFF3 + (i2-L2)*OFF2 + (i1-L1)*OFF1 + i0-L0];
    }


    template <class TLOC , typename std::enable_if<std::is_arithmetic<TLOC>::value,int>::type = 0 >
    YAKL_INLINE void operator= (TLOC val) { for (int i=0 ; i < totElems() ; i++) { myData[i] = val; } }


    YAKL_INLINE T *data    () const { return myData; }
    YAKL_INLINE T *get_data() const { return myData; }
    static unsigned constexpr totElems      () { return D3*D2*D1*D0; }
    static unsigned constexpr get_totElems  () { return D3*D2*D1*D0; }
    static unsigned constexpr get_elem_count() { return D3*D2*D1*D0; }
    static unsigned constexpr get_rank      () { return rank; }
    static bool     constexpr span_is_contiguous() { return true; }
    static bool     constexpr initialized() { return true; }


    inline friend std::ostream &operator<<(std::ostream& os, FSArray const &v) {
      for (int i=0; i<totElems(); i++) { os << std::setw(12) << v.myData[i] << "\n"; }
      os << "\n";
      return os;
    }

    
    YAKL_INLINE FSArray<int,1,SB<rank>> get_dimensions() const {
      FSArray<int,1,SB<rank>> ret;
      if constexpr (rank >= 1) ret(1) = D0;
      if constexpr (rank >= 2) ret(2) = D1;
      if constexpr (rank >= 3) ret(3) = D2;
      if constexpr (rank >= 4) ret(4) = D3;
      return ret;
    }
    YAKL_INLINE FSArray<int,1,SB<rank>> get_lbounds() const {
      FSArray<int,1,SB<rank>> ret;
      if constexpr (rank >= 1) ret(1) = L0;
      if constexpr (rank >= 2) ret(2) = L1;
      if constexpr (rank >= 3) ret(3) = L2;
      if constexpr (rank >= 4) ret(4) = L3;
      return ret;
    }
    YAKL_INLINE FSArray<int,1,SB<rank>> get_ubounds() const {
      FSArray<int,1,SB<rank>> ret;
      if constexpr (rank >= 1) ret(1) = U0;
      if constexpr (rank >= 2) ret(2) = U1;
      if constexpr (rank >= 3) ret(3) = U2;
      if constexpr (rank >= 4) ret(4) = U3;
      return ret;
    }

  };

}


