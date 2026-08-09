
#pragma once
// Included by YAKL.h


#include <chrono>
#include <cerrno>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
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
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <optional>
#include "stdlib.h"
#include <list>
#include <functional>
#include <stdexcept>
#include <sstream>
#include <string>

#ifndef YAKL_INDEX_BITS
  #define YAKL_INDEX_BITS 64
#endif

namespace yakl {
  #if YAKL_INDEX_BITS == 32
    using index_t  = std::int32_t;
    using uindex_t = std::uint32_t;
  #elif YAKL_INDEX_BITS == 64
    using index_t  = std::int64_t;
    using uindex_t = std::uint64_t;
  #else
    #error "YAKL_INDEX_BITS must be 32 or 64"
  #endif

  int constexpr index_bits = YAKL_INDEX_BITS;
}

#ifdef HAVE_MPI
#include <mpi.h>
#endif
