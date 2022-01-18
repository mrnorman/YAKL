
#pragma once 

#include <iostream>
#include <iomanip>
#include <time.h>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <memory>
#include <cstring>
#include <vector>
#include <mutex>
#include "stdlib.h"
#include <list>
#include <functional>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef YAKL_DEBUG
#include <stdexcept>
#include <sstream>
#include <string>
#endif

#ifdef _OPENMP45
#include <omp.h>
#endif

#ifdef _OPENACC
#include "openacc.h"
#endif

#if   defined(YAKL_ARCH_CUDA)
  #include <cub/cub.cuh>
#elif defined(YAKL_ARCH_HIP)
  #include "hip/hip_runtime.h"
  #include <hipcub/hipcub.hpp>
#elif defined(YAKL_ARCH_SYCL)
  #include <CL/sycl.hpp>
#elif defined(YAKL_ARCH_OPENMP45)
  #include <omp.h>
#elif defined(YAKL_ARCH_OPENMP)
  #include <omp.h>
#endif

#ifdef YAKL_ARCH_CUDA
  #include <nvtx3/nvToolsExt.h>
#endif

