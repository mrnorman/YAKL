
#include <iostream>
#include "YAKL.h"
#include "FArray.h"

using yakl::COLON;

int main() {
  yakl::init();
  yakl::FArray<float,yakl::memHost> large("large",{0,50},{-2,20},{-1,60},{-2,80});

  {
    auto slice = large.slice(COLON,COLON,43,52);
    std::cout << &(large(0,-2,43,52)) << std::endl;
    std::cout << slice.data() << std::endl;
  }

  {
    auto slice = large.slice(COLON,COLON,60,81);
    std::cout << &(large(0,-2,60,81)) << std::endl;
    std::cout << slice.data() << std::endl;
  }

  yakl::finalize();
}

