
#pragma once
// Included by YAKL_Array.h

namespace yakl {

  struct Bnds { index_t l, u; };

  KOKKOS_INLINE_FUNCTION constexpr uindex_t bnds_extent(Bnds bnd) {
    return static_cast<uindex_t>(static_cast<uindex_t>(bnd.u)-static_cast<uindex_t>(bnd.l)) + 1;
  }

  template <Bnds... DIMS>
  consteval bool valid_bnds_product() {
    uindex_t product = 1;
    for (uindex_t extent : {bnds_extent(DIMS)...}) {
      if (extent == 0 || product > std::numeric_limits<uindex_t>::max()/extent) return false;
      product *= extent;
    }
    return true;
  }



  template <class T, Bnds... DIMS> requires (sizeof...(DIMS) > 0) &&
                                            ((DIMS.l <= DIMS.u) && ...) &&
                                            ((bnds_extent(DIMS) > 0) && ...) &&
                                            (valid_bnds_product<DIMS...>())
  class SArray_F {
    public:
    bool         static constexpr is_SArray    = true;
    unsigned int static constexpr rank         = sizeof...(DIMS);
    uindex_t     static constexpr num_elements = (bnds_extent(DIMS) * ...);
    bool         static constexpr is_cstyle    = false;
    bool         static constexpr is_fstyle    = true;
    using value_type           = T;
    using const_value_type     = std::add_const_t<T>;
    using non_const_value_type = std::remove_cv_t<T>;

    T mutable my_data[num_elements];

    template <class TLOC> requires std::is_arithmetic_v<TLOC>
    KOKKOS_INLINE_FUNCTION void operator= (TLOC val) { for (uindex_t i=0; i < size(); i++) { my_data[i] = val; } }

    KOKKOS_INLINE_FUNCTION T & operator()(std::integral auto... indices) const {
      index_t  constexpr lb  [rank] = {DIMS.l...};
      index_t  constexpr ub  [rank] = {DIMS.u...};
      uindex_t constexpr dims[rank] = {bnds_extent(DIMS)...};
      std::array<uindex_t,rank> constexpr offsets = [=] {
        std::array<uindex_t,rank> result = {};
        for (int i=static_cast<int>(rank)-1; i >= 0; i--) {
          result[i] = 1;
          for (int j = i-1; j >= 0; j--) result[i] *= dims[j];
        }
        return result;
      }();
      static_assert( sizeof...(indices) == rank , "ERROR: Indexing SArray_F with the wrong number of indices" );
      index_t idx[rank] = {checked_index(indices,"ERROR: SArray_F index is not representable by index_t")...};
      if constexpr (kokkos_bounds_debug) {
        for (int i = 0; i < rank; i++) {
          if (idx[i] > ub[i] || idx[i] < lb[i]) Kokkos::abort("ERROR: SArray_F index out of bounds");
        }
      }
      uindex_t offset = 0;
      for (int i = 0; i < rank; i++) offset += static_cast<uindex_t>(idx[i]-lb[i]) * offsets[i];
      return my_data[offset];
    }

    KOKKOS_INLINE_FUNCTION T * data () const { return my_data; }
    KOKKOS_INLINE_FUNCTION T * begin() const { return my_data; }
    KOKKOS_INLINE_FUNCTION T * end  () const { return my_data + size(); }
    KOKKOS_INLINE_FUNCTION uindex_t     static constexpr size() { return num_elements; }
    KOKKOS_INLINE_FUNCTION bool         static constexpr span_is_contiguous() { return true; }
    KOKKOS_INLINE_FUNCTION bool         static constexpr is_allocated() { return true; }
    KOKKOS_INLINE_FUNCTION uindex_t     static           extent(std::integral auto i) {
      uindex_t constexpr dims[rank] = {bnds_extent(DIMS)...};
      if constexpr (kokkos_debug) {
        if ((std::is_signed_v<decltype(i)> && i < 0) || static_cast<unsigned int>(i) >= rank) {
          Kokkos::abort("ERROR: calling SArray_F extent() with out of bounds index"); 
        }
      }
      return dims[i];
    }
    template <std::integral auto I> requires (I >=0 ) && (I < rank)
    KOKKOS_INLINE_FUNCTION uindex_t     static constexpr extent() {
      uindex_t constexpr dims[rank] = {bnds_extent(DIMS)...};
      return dims[I];
    }

    inline friend std::ostream &operator<<( std::ostream& os , SArray_F const & v ) {
      os << "yakl::SArray_F: ";
      for (int i = 0; i < size(); i++) { os << v.my_data[i] << (i<size()-1 ? " , " : ""); }
      os << std::endl;
      return os;
    }

    KOKKOS_INLINE_FUNCTION auto extents() const {
      uindex_t constexpr dims[rank] = {bnds_extent(DIMS)...};
      SArray_F<uindex_t,Bnds{1,rank}> ret;
      for (int i=1; i <= rank; i++) { ret(i) = dims[i-1]; }
      return ret;
    }

    KOKKOS_INLINE_FUNCTION auto lbounds() const {
      index_t constexpr lb[rank] = {DIMS.l...};
      SArray_F<index_t,Bnds{1,rank}> ret;
      for (int i=1; i <= rank; i++) { ret(i) = lb[i-1]; }
      return ret;
    }

    KOKKOS_INLINE_FUNCTION auto ubounds() const {
      index_t constexpr ub[rank] = {DIMS.u...};
      SArray_F<index_t,Bnds{1,rank}> ret;
      for (int i=1; i <= rank; i++) { ret(i) = ub[i-1]; }
      return ret;
    }

    KOKKOS_INLINE_FUNCTION auto unpack_global_index(std::integral auto iglob) const {
      index_t constexpr lb[rank] = {DIMS.l...};
      uindex_t constexpr dims[rank] = {bnds_extent(DIMS)...};
      std::array<uindex_t,rank> constexpr offsets = [=] {
        std::array<uindex_t,rank> result = {};
        for (int i=static_cast<int>(rank)-1; i >= 0; i--) {
          result[i] = 1;
          for (int j = i-1; j >= 0; j--) result[i] *= dims[j];
        }
        return result;
      }();
      SArray_F<index_t,Bnds{1,rank}> ret;
      if constexpr (kokkos_bounds_debug) {
        if ((std::is_signed_v<decltype(iglob)> && iglob < 0) || static_cast<uindex_t>(iglob) >= size()) {
          Kokkos::abort("ERROR: SArray_F::unpack_global_index index out of bounds");
        }
      }
      for (int i=1; i <= rank; i++) {
        ret(i) = static_cast<index_t>((static_cast<uindex_t>(iglob) / offsets[i-1]) % dims[i-1]) + lb[i-1];
      }
      return ret;
    }

    template <class NEW> using TypeAs = SArray_F<NEW,DIMS...>;
  };

}
