
#pragma once

#include "YAKL_header.h"
#include "YAKL_defines.h"

/** @namespace yakl::c
  * @brief Contains `Bounds` class, and `parallel_for()` routines using C-style indexing and ordering */

/** @namespace yakl::fortran
  * @brief Contains `Bounds` class, and `parallel_for()` routines using Fortran-style indexing and ordering */

namespace yakl {
  using std::cos;
  using std::sin;
  using std::pow;
  using std::min;
  using std::max;
  using std::abs;

  // functorBuffer holds large functors in CUDA and all functors in SYCL
  /** @private */
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


/** @mainpage Yet Another Kernel Launcher (YAKL) API Documentation
 *
 * Welcome to the full API documentation for YAKL. Here you will find detailed documentation for all user-facing
 * functions and classes in YAKL. For a more tutorial style introduction, please visit: https://github.com/mrnorman/YAKL/wiki
 *
 * Not sure where to begin? Try looking at the following links:
 *
 * ## Namespaces
 * <a href="https://mrnorman.github.io/yakl_api/html/namespaces.html">Namespaces</a> (A list of all functions and classes in each YAKL namespace)
 *
 * In each namespace, you'll see a list of classes, types, functions, and variables.
 * Each of these has a brief description, and clicking on the name will typically give more information
 * about that class, type, function, or variable.
 *
 * * <a href="https://mrnorman.github.io/yakl_api/html/namespaceyakl.html">yakl</a>
 *   - <a href="https://mrnorman.github.io/yakl_api/html/namespaceyakl_1_1c.html">c</a>
 *   - <a href="https://mrnorman.github.io/yakl_api/html/namespaceyakl_1_1componentwise.html">components</a>
 *   - <a href="https://mrnorman.github.io/yakl_api/html/namespaceyakl_1_1fortran.html">fortran</a>
 *   - <a href="https://mrnorman.github.io/yakl_api/html/namespaceyakl_1_1intrinsics.html">intrinsics</a>
 *
 * ## Classes
 * <a href="https://mrnorman.github.io/yakl_api/html/annotated.html">Classes</a> (A list of all classes in YAKL)
 */


