
#include <iostream>
#include "Array.h"


int const vectorSize = 128;


#ifdef __NVCC__
  template <class F> __global__ void cudaKernel(ulong const nIter, F f) {
    ulong i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < nIter) {
      f( i );
    }
  }
#endif


template <class F> void parallelFor( int const nIter , F f ) {
  #ifdef __NVCC__
    cudaKernel <<< (uint) nIter/vectorSize+1 , vectorSize >>> ( nIter , f );
  #else
    for (int i=0; i<nIter; i++) {
      f(i);
    }
  #endif
}


int main() {
  int n = 1024*1024;
  Array<float> a, b, c;
  a = Array<float>("a",n);
  b = Array<float>("b",n);
  c = Array<float>("c",n);
  a = 2;
  b = 3;
  parallelFor( n , [=] _HOSTDEV (int i) { c(i) = a(i) + b(i); } );
  // launcher.parallelFor( n , [=] __host__ __device__ (int i) { c(i) = 0.; } );
  // synchronizeSelf();
  #ifdef __NVCC__
  cudaDeviceSynchronize();
  #endif
  std::cout << c.sum() / 1024 / 1024 << "\n";
}
