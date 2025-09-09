
#pragma once
// Included by YAKL.h

#ifdef YAKL_AUTO_PROFILE
  #ifndef YAKL_PROFILE
    #define YAKL_PROFILE
  #endif
#endif

#ifdef KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
  #ifndef KOKKOS_ENABLE_DEBUG
    #define KOKKOS_ENABLE_DEBUG
  #endif
#endif

namespace yakl {
  inline std::string my_basename(const std::string& path) {
      size_t last_slash = path.find_last_of("/\\");
      if (std::string::npos == last_slash) {
          return path;
      }
      return path.substr(last_slash + 1);
  }
}


#define YAKL_AUTO_LABEL() (yakl::my_basename(__FILE__) + std::string(":") + std::to_string(__LINE__)).c_str()
#define YAKL_SCOPE(a,b) auto &a = std::ref(b).get()

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
