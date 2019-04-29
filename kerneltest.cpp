
#include "const.h"
#include "Array.h"
#include <iostream>
#include "YAKL.h"


int main() {
  Array<float> a, b, c;
  int const n = 1024*1024;
  a.setup(n);
  b.setup(n);
  c.setup(n);

  a = 2;
  b = 3;

  yakl::Launcher<yakl::targetCUDA> launcher;

  launcher.parallelFor( n , [] _YAKL (unsigned long i, Array<float> &a, Array<float> &b, Array<float> &c ) { c(i) = a(i) + b(i); } , a , b , c );
  launcher.synchronize();

  std::cout << (int) c.sum() << " " << 5*1024*1024 << "\n";

}



