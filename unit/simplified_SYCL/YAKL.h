
#pragma once

#include <iostream>
#include <iomanip>
#include <time.h>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cstring>
#include <vector>
#include "stdlib.h"

#define YAKL_LAMBDA [=]
#define YAKL_INLINE inline
#define YAKL_DEVICE inline
#define YAKL_SCOPE(a,b) auto &a = std::ref(b).get()
#include <CL/sycl.hpp>

namespace yakl {

  typedef unsigned int index_t;

  extern sycl::queue sycl_default_stream;

  // Memory space specifiers for YAKL Arrays
  int constexpr memDevice = 1;
  int constexpr memHost   = 2;
  int constexpr memStack  = 3;
  int constexpr memDefault = memDevice;


  int constexpr styleC       = 1;
  int constexpr styleFortran = 2;
  int constexpr styleDefault = styleC;


  extern bool yakl_is_initialized;

  inline void init() {
    yakl_is_initialized = true;
  }

  inline void finalize() {
    yakl_is_initialized = false;
    sycl_default_stream = sycl::queue();
  }


#include "YAKL_parallel_for_c.h"

#include "Array.h"

}



