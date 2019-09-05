
#include "stdlib.h"
#include <iostream>
#include <string>
#include "Array.h"
#include "SArray.h"
#include "YAKL.h"


// To compile on the CPU:
// g++ -DARRAY_DEBUG vla_kernel.cpp

// To compile on the GPU:
// nvcc -x cu --expt-extended-lambda --expt-relaxed-constexpr vla_kernel.cpp


int main() {
  int vlen;
  std::cout << "Size of the VLA: ";
  std::cin >> vlen;
  int const n = 1024*1024;
  Array<float> c("c",n);
  
  yakl::parallel_for( n , [=] _YAKL (int const j) {
    float tmp[vlen];
    for (int i=0; i<vlen; i++) {
      tmp[i] = i*j;
    }
    float sm = 0;
    for (int i=0; i<vlen; i++) {
      sm += tmp[i];
    }
    c(j) = sm;
  });

  yakl::fence();

  std::cout << c(n-1) << "\n";

}

