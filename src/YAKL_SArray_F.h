
#pragma once
// Included by YAKL_Array.h

namespace yakl {

  template <class T, Bnds... DIMS> requires (sizeof...(DIMS) > 0) &&
                                            ((static_cast<int>(DIMS.l) <= static_cast<int>(DIMS.u)) && ...)
  class SArray_F {
    public:
    using style = FStyle;
    bool                          static constexpr is_SArray    = true;
    unsigned int                  static constexpr rank         = sizeof...(DIMS);
    int                           static constexpr lb  [rank]   = {static_cast<int>(DIMS.l)...};
    int                           static constexpr ub  [rank]   = {static_cast<int>(DIMS.u)...};
    unsigned int                  static constexpr dims[rank]   = {(static_cast<unsigned int>(static_cast<int>(DIMS.u)-static_cast<int>(DIMS.l)+1))...};
    unsigned int                  static constexpr num_elements = ((static_cast<unsigned int>(static_cast<int>(DIMS.u)-static_cast<int>(DIMS.l)+1)) * ...);
    bool                          static constexpr is_cstyle    = false;
    bool                          static constexpr is_fstyle    = true;
    std::array<unsigned int,rank> static constexpr offsets      = [] {
      std::array<unsigned int,rank> result = {};
      for (int i=static_cast<int>(rank)-1; i >= 0; i--) {
        result[i] = 1;
        for (int j = i-1; j >= 0; j--) result[i] *= dims[j];
      }
      return result;
    }();
    using value_type           = T;
    using const_value_type     = std::add_const_t<T>;
    using non_const_value_type = std::remove_cv_t<T>;

    T mutable my_data[num_elements];

    template <class TLOC> requires std::is_arithmetic_v<TLOC>
    KOKKOS_INLINE_FUNCTION void operator= (TLOC val) { for (unsigned int i=0; i < size(); i++) { my_data[i] = val; } }

    KOKKOS_INLINE_FUNCTION T & operator()(std::integral auto... indices) const {
      static_assert( sizeof...(indices) == rank , "ERROR: Indexing SArray_F with the wrong number of indices" );
      int idx[rank] = {static_cast<int>(indices)...};
      unsigned int offset = 0;
      for (int i = 0; i < rank; i++) offset += (idx[i]-lb[i]) * offsets[i];
      if constexpr (kokkos_bounds_debug) {
        for (int i = 0; i < rank; i++) {
          if (idx[i] > ub[i] || idx[i] < lb[i]) Kokkos::abort("ERROR: SArray_F index out of bounds");
        }
      }
      return my_data[offset];
    }

    KOKKOS_INLINE_FUNCTION T * data () const { return my_data; }
    KOKKOS_INLINE_FUNCTION T * begin() const { return my_data; }
    KOKKOS_INLINE_FUNCTION T * end  () const { return my_data + size(); }
    KOKKOS_INLINE_FUNCTION unsigned int static constexpr size() { return num_elements; }
    KOKKOS_INLINE_FUNCTION bool   static constexpr span_is_contiguous() { return true; }
    KOKKOS_INLINE_FUNCTION bool   static constexpr is_allocated() { return true; }
    KOKKOS_INLINE_FUNCTION unsigned int static constexpr extent(std::integral auto i) {
      if constexpr (kokkos_debug) {
        if ((std::is_signed_v<decltype(i)> && i < 0) || static_cast<unsigned int>(i) >= rank) {
          Kokkos::abort("ERROR: calling SArray_F extent() with out of bounds index"); 
        }
      }
      return dims[i];
    }

    inline friend std::ostream &operator<<( std::ostream& os , SArray_F const & v ) {
      os << "yakl::SArray_F: ";
      for (unsigned int i = 0; i < size(); i++) { os << v.my_data[i] << (i<size()-1 ? " , " : ""); }
      os << std::endl;
      return os;
    }

    KOKKOS_INLINE_FUNCTION auto extents() const {
      SArray_F<unsigned int,{1,rank}> ret;
      for (unsigned int i=1; i <= rank; i++) { ret(i) = dims[i-1]; }
      return ret;
    }

    KOKKOS_INLINE_FUNCTION auto lbounds() const {
      SArray_F<int,{1,rank}> ret;
      for (unsigned int i=1; i <= rank; i++) { ret(i) = lb[i-1]; }
      return ret;
    }

    KOKKOS_INLINE_FUNCTION auto ubounds() const {
      SArray_F<int,{1,rank}> ret;
      for (unsigned int i=1; i <= rank; i++) { ret(i) = ub[i-1]; }
      return ret;
    }

    KOKKOS_INLINE_FUNCTION auto unpack_global_index(std::integral auto iglob) const {
      SArray_F<int,{1,rank}> ret;
      for (int i=1; i <= rank; i++) { ret(i) = iglob / offsets[i-1] + lb[i-1]; }
      return ret;
    }
  };

}


