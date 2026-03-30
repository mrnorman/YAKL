
#pragma once
// Included by YAKL_Array.h

namespace yakl {

  template <class Type> inline constexpr bool is_SArray = requires { Type::is_SArray; };



  template <class T, std::integral auto... DIMS> requires (sizeof...(DIMS) > 0) &&
                                                          ((DIMS > 0) && ...)
  class SArray {
    public:
    using style = CStyle;
    bool                          static constexpr is_SArray    = true;
    unsigned int                  static constexpr rank         = sizeof...(DIMS);
    unsigned int                  static constexpr dims[rank]   = {DIMS...};
    unsigned int                  static constexpr num_elements = (DIMS * ...);
    bool                          static constexpr is_cstyle    = true;
    bool                          static constexpr is_fstyle    = false;
    std::array<unsigned int,rank> static constexpr offsets      = [] {
      std::array<unsigned int,rank> result = {};
      for (int i=0; i < static_cast<int>(rank); i++) {
        result[i] = 1;
        for (int j = i+1; j < static_cast<int>(rank); j++) result[i] *= dims[j];
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
      static_assert( sizeof...(indices) == rank , "ERROR: Indexing SArray with the wrong number of indices" );
      unsigned int idx[rank] = {static_cast<unsigned int>(indices)...};
      unsigned int offset = 0;
      for (int i = 0; i < rank; i++) offset += idx[i] * offsets[i];
      if constexpr (kokkos_bounds_debug) {
        for (int i = 0; i < rank; i++) {
          if (idx[i] >= dims[i]) Kokkos::abort("ERROR: SArray index out of bounds");
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
          Kokkos::abort("ERROR: calling SArray extent() with out of bounds index"); 
        }
      }
      return dims[i];
    }

    inline friend std::ostream &operator<<( std::ostream& os , SArray const & v ) {
      os << "yakl::SArray: ";
      for (unsigned int i = 0; i < size(); i++) { os << v.my_data[i] << (i<size()-1 ? " , " : ""); }
      os << std::endl;
      return os;
    }

    KOKKOS_INLINE_FUNCTION auto extents() const {
      SArray<unsigned int,rank> ret;
      for (unsigned int i=0; i < rank; i++) { ret(i) = dims[i]; }
      return ret;
    }

    KOKKOS_INLINE_FUNCTION auto lbounds() const {
      SArray<unsigned int,rank> ret;
      for (unsigned int i=0; i < rank; i++) { ret(i) = 0; }
      return ret;
    }

    KOKKOS_INLINE_FUNCTION auto ubounds() const {
      SArray<unsigned int,rank> ret;
      for (unsigned int i=0; i < rank; i++) { ret(i) = dims[i]-1; }
      return ret;
    }

    KOKKOS_INLINE_FUNCTION auto unpack_global_index(std::integral auto iglob) const {
      SArray<unsigned int,rank> ret;
      for (int i=0; i < rank; i++) { ret(i) = iglob / offsets[i]; }
      return ret;
    }

    template <class NEW> using TypeAs = SArray<NEW,DIMS...>;
  };

}


