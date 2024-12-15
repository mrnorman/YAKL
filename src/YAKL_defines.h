
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

#define YAKL_AUTO_LABEL() (std::string(basename(__FILE__)) + std::string(":") + std::to_string(__LINE__)).c_str()
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
