
#pragma once
// Included by YAKL.h

#ifdef KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
  #ifndef KOKKOS_ENABLE_DEBUG
    #define KOKKOS_ENABLE_DEBUG
  #endif
#endif

namespace yakl {
  #ifdef KOKKOS_ENABLE_DEBUG
    inline constexpr bool kokkos_debug = true;
  #else
    inline constexpr bool kokkos_debug = false;
  #endif

  #ifdef KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
    inline constexpr bool kokkos_bounds_debug = true;
  #else
    inline constexpr bool kokkos_bounds_debug = false;
  #endif

  #ifdef YAKL_AUTO_FENCE
    inline constexpr bool yakl_auto_fence = true;
  #else
    inline constexpr bool yakl_auto_fence = false;
  #endif

  #ifdef HAVE_MPI
    inline constexpr bool have_mpi = true;
  #else
    inline constexpr bool have_mpi = false;
  #endif

  inline std::string my_basename(const std::string& path) {
      size_t last_slash = path.find_last_of("/\\");
      if (std::string::npos == last_slash) {
          return path;
      }
      return path.substr(last_slash + 1);
  }
}


// #define YAKL_AUTO_LABEL() (yakl::my_basename(__FILE__) + std::string(":") + std::to_string(__LINE__)).c_str()
#define YAKL_AUTO_LABEL() ""
#if defined(KOKKOS_ENABLE_HIP)
#define YAKL_SCOPE(a,b) auto &a = b
#elif defined(KOKKOS_ENABLE_CUDA)
#define YAKL_SCOPE(a,b) auto a = b
#else
#define YAKL_SCOPE(a,b) auto &a = std::ref(b).get()
#endif

namespace yakl {
  #ifdef HAVE_MPI 
    inline bool yakl_mainproc(MPI_Comm comm = MPI_COMM_WORLD) {
      int init;
      MPI_Initialized( &init );
      if (init) {
        int rank;
        MPI_Comm_rank( comm , &rank );
        return rank == 0;
      }
      return true;
    }
  #else
    inline bool yakl_mainproc() { return true; }
  #endif
}
