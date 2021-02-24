
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
namespace sycl = cl::sycl;

//Contents added to namespace below
//#include "YAKL_alloc_free"

namespace yakl {

  typedef unsigned int index_t;

  extern sycl::queue sycl_default_stream;

  // Memory space specifiers for YAKL Arrays
  int constexpr memDevice = 1;
  int constexpr memHost   = 2;
  int constexpr memStack  = 3;
  #if defined(__USE_CUDA__) || defined(__USE_HIP__) || defined(__USE_SYCL__)
    int constexpr memDefault = memDevice;
  #else
    int constexpr memDefault = memHost;
  #endif


  int constexpr styleC       = 1;
  int constexpr styleFortran = 2;
  int constexpr styleDefault = styleC;


  extern bool yakl_is_initialized;


#include "YAKL_init.h"


  inline void finalize() {
    yakl_is_initialized = false;
    sycl_default_stream = sycl::queue();
  }


#include "YAKL_parallel_for.h"

#include "Array.h"

}



