
#pragma once
// Included by YAKL_Array.h

namespace yakl {

  template <class Type> inline constexpr bool is_SArray = requires { requires Type::is_SArray; };



  template <std::integral auto... DIMS>
  consteval bool valid_sarray_product() {
    size_t product = 1;
    for (size_t extent : {static_cast<size_t>(DIMS)...}) {
      if (extent == 0 || product > std::numeric_limits<size_t>::max()/extent) return false;
      product *= extent;
    }
    return true;
  }



  template <class T, std::integral auto... DIMS> requires (sizeof...(DIMS) > 0) &&
                                                          ((DIMS > 0) && ...) &&
                                                          ((std::in_range<size_t>(DIMS)) && ...) &&
                                                          (valid_sarray_product<DIMS...>())
  class SArray {
    public:
    bool         static constexpr is_SArray    = true;
    unsigned int static constexpr rank         = sizeof...(DIMS);
    size_t       static constexpr num_elements = (static_cast<size_t>(DIMS) * ...);
    bool         static constexpr is_cstyle    = true;
    bool         static constexpr is_fstyle    = false;
    using value_type           = T;
    using const_value_type     = std::add_const_t<T>;
    using non_const_value_type = std::remove_cv_t<T>;

    T mutable my_data[num_elements];

    template <class TLOC> requires std::is_arithmetic_v<TLOC>
    KOKKOS_INLINE_FUNCTION void operator= (TLOC val) { for (size_t i=0; i < size(); i++) { my_data[i] = val; } }

    KOKKOS_INLINE_FUNCTION T & operator()(std::integral auto... indices) const {
      static_assert( sizeof...(indices) == rank , "ERROR: Indexing SArray with the wrong number of indices" );
      size_t idx[rank] = {static_cast<size_t>(indices)...};
      size_t constexpr dims[rank] = {static_cast<size_t>(DIMS)...};
      std::array<size_t,rank> constexpr offsets = [=] {
        std::array<size_t,rank> result = {};
        for (int i=0; i < static_cast<int>(rank); i++) {
          result[i] = 1;
          for (int j = i+1; j < static_cast<int>(rank); j++) result[i] *= dims[j];
        }
        return result;
      }();
      if constexpr (kokkos_bounds_debug) {
        for (int i = 0; i < rank; i++) {
          if (idx[i] >= dims[i]) Kokkos::abort("ERROR: SArray index out of bounds");
        }
      }
      size_t offset = 0;
      for (int i = 0; i < rank; i++) offset += idx[i] * offsets[i];
      return my_data[offset];
    }

    KOKKOS_INLINE_FUNCTION T * data () const { return my_data; }
    KOKKOS_INLINE_FUNCTION T * begin() const { return my_data; }
    KOKKOS_INLINE_FUNCTION T * end  () const { return my_data + size(); }
    KOKKOS_INLINE_FUNCTION size_t       static constexpr size() { return num_elements; }
    KOKKOS_INLINE_FUNCTION bool         static constexpr span_is_contiguous() { return true; }
    KOKKOS_INLINE_FUNCTION bool         static constexpr is_allocated() { return true; }
    KOKKOS_INLINE_FUNCTION size_t       static           extent(std::integral auto i) {
      size_t constexpr dims[rank] = {static_cast<size_t>(DIMS)...};
      if constexpr (kokkos_debug) {
        if ((std::is_signed_v<decltype(i)> && i < 0) || static_cast<unsigned int>(i) >= rank) {
          Kokkos::abort("ERROR: calling SArray extent() with out of bounds index"); 
        }
      }
      return dims[i];
    }
    template <std::integral auto I> requires (I >=0 ) && (I < rank)
    KOKKOS_INLINE_FUNCTION size_t       static constexpr extent() {
      size_t constexpr dims[rank] = {static_cast<size_t>(DIMS)...};
      return dims[I];
    }

    inline friend std::ostream &operator<<( std::ostream& os , SArray const & v ) {
      os << "yakl::SArray: ";
      for (size_t i = 0; i < size(); i++) { os << v.my_data[i] << (i<size()-1 ? " , " : ""); }
      os << std::endl;
      return os;
    }

    KOKKOS_INLINE_FUNCTION auto extents() const {
      size_t constexpr dims[rank] = {static_cast<size_t>(DIMS)...};
      SArray<size_t,rank> ret;
      for (int i=0; i < rank; i++) { ret(i) = dims[i]; }
      return ret;
    }

    KOKKOS_INLINE_FUNCTION auto lbounds() const {
      SArray<size_t,rank> ret;
      for (int i=0; i < rank; i++) { ret(i) = 0; }
      return ret;
    }

    KOKKOS_INLINE_FUNCTION auto ubounds() const {
      size_t constexpr dims[rank] = {static_cast<size_t>(DIMS)...};
      SArray<size_t,rank> ret;
      for (int i=0; i < rank; i++) { ret(i) = dims[i]-1; }
      return ret;
    }

    KOKKOS_INLINE_FUNCTION auto unpack_global_index(std::integral auto iglob) const {
      size_t constexpr dims[rank] = {static_cast<size_t>(DIMS)...};
      std::array<size_t,rank> constexpr offsets = [=] {
        std::array<size_t,rank> result = {};
        for (int i=0; i < static_cast<int>(rank); i++) {
          result[i] = 1;
          for (int j = i+1; j < static_cast<int>(rank); j++) result[i] *= dims[j];
        }
        return result;
      }();
      SArray<size_t,rank> ret;
      if constexpr (kokkos_bounds_debug) {
        if ((std::is_signed_v<decltype(iglob)> && iglob < 0) || static_cast<size_t>(iglob) >= size()) {
          Kokkos::abort("ERROR: SArray::unpack_global_index index out of bounds");
        }
      }
      for (int i=0; i < rank; i++) {
        ret(i) = (static_cast<size_t>(iglob) / offsets[i]) % dims[i];
      }
      return ret;
    }

    template <class NEW> using TypeAs = SArray<NEW,DIMS...>;
  };

}
