
#pragma once

YAKL_INLINE void yakl_throw(const char * msg) {
  #ifndef YAKL_SEPARATE_MEMORY_SPACE
    std::cerr << "YAKL FATAL ERROR:\n";
    std::cerr << msg << std::endl;
    throw msg;
  #else
    #ifdef YAKL_ARCH_SYCL
      const CL_CONSTANT char format[] = "KERNEL CHECK FAILED:\n   %s\n";
      sycl::ext::oneapi::experimental::printf(format,msg);
    #else
      printf("%s\n",msg);
    #endif
    // Intentionally cause a segfault to kill the run if you're on a GPU
    int *segfault = nullptr;
    *segfault = 10;
  #endif
}


inline void check_last_error() {
  #ifdef YAKL_DEBUG
    #ifdef YAKL_ARCH_CUDA
      auto ierr = cudaGetLastError();
      if (ierr != cudaSuccess) { yakl_throw( cudaGetErrorString( ierr ) ); }
    #elif defined(YAKL_ARCH_HIP)
      auto ierr = hipGetLastError();
      if (ierr != hipSuccess) { yakl_throw( hipGetErrorString( ierr ) ); }
    #elif defined(YAKL_ARCH_SYCL)
    #elif defined(YAKL_ARCH_OPENMP45)
      //auto ierr = GetLastError();
    #endif
  #endif
}

inline bool yakl_masterproc() {
  #ifdef HAVE_MPI
    int is_initialized;
    MPI_Initialized(&is_initialized);
    if (!is_initialized) {
      return true;
    } else {
      int myrank;
      MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
      return myrank == 0;
    }
  #else
    return true;
  #endif
}


