
#pragma once

#include "YAKL_header.h"
#include "YAKL_defines.h"

namespace yakl {
  using std::cos;
  using std::sin;
  using std::pow;
  using std::min;
  using std::max;
  using std::abs;

  // functorBuffer holds large functors in CUDA and all functors in SYCL
  int constexpr functorBufSize = 1024*128;
  extern void *functorBuffer;

  // Type for indexing. Rarely if ever is size_t going to be needed
  typedef unsigned int index_t;
  index_t constexpr INDEX_MAX = std::numeric_limits<index_t>::max();
}

#include "YAKL_mutex.h"
#include "YAKL_sycldevice.h"
#include "YAKL_LaunchConfig.h"
#include "YAKL_fence.h"
#include "YAKL_error.h"
#include "YAKL_simd.h"
#include "YAKL_Gator.h"
#include "YAKL_allocators.h"
#include "YAKL_timers.h"
#include "YAKL_InitConfig.h"
#include "YAKL_init.h"
#include "YAKL_finalize.h"
#include "YAKL_parallel_for.h"
#include "YAKL_memory_spaces.h"
#include "YAKL_mem_transfers.h"
#include "YAKL_reductions.h"
#include "YAKL_atomics.h"
#include "YAKL_random.h"
#include "YAKL_Array.h"
#include "YAKL_ScalarLiveOut.h"
#include "extensions/YAKL_componentwise.h"
#include "extensions/YAKL_intrinsics.h"
#include "YAKL_memset.h"
#include "extensions/YAKL_fft.h"

