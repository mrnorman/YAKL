
#pragma once

namespace yakl {

  inline void yakl_throw(std::string exc) {
    std::cout << "YAKL FATAL ERROR:\n";
    std::cout << exc << std::endl;
    throw exc;
  }


  inline void check_last_error() {
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
  }

}


