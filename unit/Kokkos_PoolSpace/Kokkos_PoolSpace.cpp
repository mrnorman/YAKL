
#include "YAKL.h"

void die(std::string msg) {
  Kokkos::abort(msg.c_str());
}


int main() {
  Kokkos::initialize();
  yakl::init();
  {
    Kokkos::View<float *,Kokkos::LayoutRight,yakl::PoolSpace> arr("arr",10);
  }
  yakl::finalize();
  Kokkos::finalize(); 
  return 0;
}

