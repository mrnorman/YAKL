
#pragma once

#include "YAKL_defines.h"

namespace yakl {

  YAKL_INLINE void yakl_throw(const char * msg) {
    #ifndef YAKL_SEPARATE_MEMORY_SPACE
      std::cerr << "YAKL FATAL ERROR:\n";
      std::cerr << msg << std::endl;
      throw msg;
    #else
      printf(msg);
      printf("\n");
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

}


