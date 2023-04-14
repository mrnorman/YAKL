
#pragma once 
// Included by YAKL.h


#include <chrono>
#include <string>
#include <unordered_map>
#include <thread>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <time.h>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <memory>
#include <cstring>
#include <vector>
#include <array>
#include <mutex>
#include "stdlib.h"
#include <list>
#include <functional>
#include <stdexcept>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef YAKL_DEBUG
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
  #include "cufft.h"
#elif defined(YAKL_ARCH_HIP)
  #include "hip/hip_runtime.h"
  #include "hipcub/hipcub.hpp"
  #include "roctracer/roctx.h"
  #include "rocfft.h"
#elif defined(YAKL_ARCH_SYCL)
  #include <sycl/sycl.hpp>
  #include <oneapi/mkl/exceptions.hpp>
  #include <oneapi/mkl/dfti.hpp>
#elif defined(YAKL_ARCH_OPENMP)
  #include <omp.h>
#endif

#ifdef YAKL_ARCH_CUDA
  #include <nvtx3/nvToolsExt.h>
#endif

