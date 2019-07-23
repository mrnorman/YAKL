
#include <iostream>
#include "Array.h"
#include "YAKL.h"


int main() {
  int n = 1024*1024;
  Array<float> a, b, c;
  a = Array<float>("a",n);
  b = Array<float>("b",n);
  c = Array<float>("c",n);
  a = 2;
  b = 3;
  yakl::parallel_for( n , [=] _HOSTDEV (int i) { c(i) = a(i) + b(i); } );
  yakl::fence();
  std::cout << c.sum() / 1024 / 1024 << "\n";
}
