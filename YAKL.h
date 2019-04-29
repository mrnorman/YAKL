
#ifndef __YAKL_H_
#define __YAKL_H_

#include <iostream>

#ifdef __NVCC__
  #define _YAKL __host__ __device__
#else
  #define _YAKL
#endif


namespace yakl {


  // These unfotunately cannot live inside classes in CUDA :-/
  #ifdef __NVCC__
  template <class F, class... Args> __global__ void parallelForCUDA(ulong const nIter, F &f, Args&&... args) {
    ulong i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < nIter) {
      f( i , args... );
    }
  }
  #endif


  int const targetCUDA      = 1;
  int const targetCPUSerial = 2;


  template <int target=targetCPUSerial, unsigned int vectorSize=128> class Launcher {
    protected:
      typedef unsigned long ulong;
      typedef unsigned int  uint;

      template <class F, class... Args> inline void parallelForCPU(ulong const nIter, F &f, Args&&... args) {
        for (ulong i=0; i<nIter; i++) {
          f( i , args... );
        }
      }

    public:

      template <class F, class... Args> inline void parallelFor(ulong const nIter, F &f, Args&&... args) {
        if        (target == targetCUDA) {
          #ifdef __NVCC__
            parallelForCUDA <<< (uint) nIter/vectorSize+1 , vectorSize >>> ( nIter , f , args... );
          #else
            std::cerr << "ERROR: "<< __FILE__ << ":" << __LINE__ << "\n";
            std::cerr << "Attempting to launch CUDA without nvcc\n";
          #endif
        } else if (target == targetCPUSerial ) {
          parallelForCPU( nIter , f , args... );
        }
      }

      template <class F, class... Args> inline void parallelFor(ulong const nIter, F const &f, Args&&... args) {
        if        (target == targetCUDA) {
          #ifdef __NVCC__
            parallelForCUDA <<< (uint) nIter/vectorSize+1 , vectorSize >>> ( nIter , f , args... );
          #else
            std::cerr << "ERROR: "<< __FILE__ << ":" << __LINE__ << "\n";
            std::cerr << "Attempting to launch CUDA without nvcc\n";
          #endif
        } else if (target == targetCPUSerial ) {
          parallelForCPU( nIter , f , args... );
        }
      }

      void synchronize() {
        if        (target == targetCUDA) {
        #ifdef __NVCC__
          cudaDeviceSynchronize();
        #endif
        } else if (target == targetCPUSerial ) {
        }
      }

  };




}


#endif

