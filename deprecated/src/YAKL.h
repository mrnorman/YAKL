
#pragma once

#include "YAKL_header.h"
#include "YAKL_defines.h"

// These wrap the yakl namespace in a user-defined namespace to allow multiple YAKLs to be used in
// the same codebase. Anonymous / unnamed namespaces are not the best choice because they would leave
// out the Fortran interoperability.
#ifdef YAKL_NAMESPACE_WRAPPER_LABEL

#define __YAKL_NAMESPACE_WRAPPER_BEGIN__ namespace YAKL_NAMESPACE_WRAPPER_LABEL {
#define __YAKL_NAMESPACE_WRAPPER_END__ }
namespace YAKL_NAMESPACE_WRAPPER_LABEL {}
using namespace YAKL_NAMESPACE_WRAPPER_LABEL ;

#else

#define __YAKL_NAMESPACE_WRAPPER_BEGIN__
#define __YAKL_NAMESPACE_WRAPPER_END__

#endif

/** @namespace yakl::c
  * @brief Contains `Bounds` class, and `parallel_for()` routines using C-style indexing and ordering */

/** @namespace yakl::fortran
  * @brief Contains `Bounds` class, and `parallel_for()` routines using Fortran-style indexing and ordering */

__YAKL_NAMESPACE_WRAPPER_BEGIN__
namespace yakl {
  using std::cos;
  using std::sin;
  using std::pow;
  using std::min;
  using std::max;
  using std::abs;

  // Type for indexing.
  typedef size_t index_t;
  index_t constexpr INDEX_MAX = std::numeric_limits<index_t>::max();
}
__YAKL_NAMESPACE_WRAPPER_END__

#include "ArrayIR.h"
#include "YAKL_verbose.h"
#include "YAKL_sycldevice.h"
#include "YAKL_streams_events.h"
#include "YAKL_LaunchConfig.h"
#include "YAKL_fence.h"
#include "YAKL_error.h"
#include "YAKL_Gator.h"
#include "YAKL_Toney.h"
#include "YAKL_Internal.h"
#include "YAKL_timers.h"
#include "YAKL_mutex.h"
#include "YAKL_allocators.h"
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
#include "YAKL_memset.h"
#include "extensions/YAKL_simd.h"
#include "extensions/YAKL_componentwise.h"
#include "extensions/YAKL_intrinsics.h"
#include "extensions/YAKL_tridiagonal.h"
#include "extensions/YAKL_pentadiagonal.h"


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
 * The <a href="https://mrnorman.github.io/yakl_api/html/namespaceyakl.html">yakl</a>, 
 * <a href="https://mrnorman.github.io/yakl_api/html/namespaceyakl_1_1c.html">c</a>, and 
 * <a href="https://mrnorman.github.io/yakl_api/html/namespaceyakl_1_1fortran.html">fortran</a> have most of the
 * functionality you will use. 
 * 
 * **Tips:**
 * * Every function decorated with `YAKL_INLINE` can be called from inside a `parallel_for()` kernel. If the function
 *   is not decorated with that, then it cannot be called from inside a `parallel_for()` kernel.
 * * If you see the `[ASYNCHRONOUS]` tag in a function's documentation, it means that function launches a `parallel_for()` kernel
 *   on the device asynchronously with respect to host code. Therefore, if you want that work to complete before
 *   performing additional host work, you must call `yakl::fence();` after the function call.
 * * If you see the `[DEEP_COPY]` tag in a function's documentation, this means a copy of all of the array object's data is being performed.
 *   If you do not see this tag, assume it is cheap to perform unless state otherwise.
 * * If you see the `[NON_B4B]` tag in a function's documentation, this means it can lead to non-bitwise reproducible results
 *   between successive runs, and you'll need to take extra steps to achieve bitwise reproducibility when you want it.
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
 *   - <a href="https://mrnorman.github.io/yakl_api/html/namespaceyakl_1_1simd.html">simd</a>
 *
 * ## Classes
 * <a href="https://mrnorman.github.io/yakl_api/html/annotated.html">Classes</a> (A list of all classes in YAKL)
 */


