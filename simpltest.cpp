
#include <iostream>
#include "Array.h"


template <class F> void parallelFor( int const numIterations , F f ) {
  #ifdef __NVCC__
    
  #else
    for (int i=0; i<numIterations; i++) {
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
  parallelFor( n , [=] (int i) { c(i) = a(i) + b(i); } );
  // launcher.parallelFor( n , [=] __host__ __device__ (int i) { c(i) = 0.; } );
  // synchronizeSelf();
  std::cout << c.sum() / 1024 / 1024 << "\n";
}
