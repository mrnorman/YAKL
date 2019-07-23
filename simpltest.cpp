
#include <iostream>
#include "YAKL.h"
#include "Array.h"


int main() {
  int n = 1024*1024;
  Array<float> a, b, c;
  a.setup(n);
  b.setup(n);
  c.setup(n);
  yakl::Launcher launcher(yakl::targetCUDA);
  a = 2;
  b = 3;
  launcher.parallelFor( n , [=] (int i) { c(i) = a(i) + b(i); } );
  // launcher.parallelFor( n , [=] __host__ __device__ (int i) { c(i) = 0.; } );
  launcher.synchronizeSelf();
  std::cout << c;
}
