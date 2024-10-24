
#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ProfileSection.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include "YAKL_header.h"
#include "YAKL_defines.h"
#include "ArrayIR.h"
#include "YAKL_LinearAllocator.h"
#include "YAKL_Toney.h"
#include "YAKL_Internal.h"
#include "YAKL_timers.h"
#include "YAKL_mutex.h"
#include "YAKL_allocators.h"
#include "YAKL_InitConfig.h"
#include "YAKL_init.h"
#include "YAKL_finalize.h"
#include "YAKL_parallel_for_c.h"
#include "YAKL_parallel_for_fortran.h"
#include "YAKL_memory_spaces.h"
#include "YAKL_memcpy.h"
#include "YAKL_random.h"
#include "YAKL_Array.h"
#include "YAKL_ScalarLiveOut.h"
#include "extensions/YAKL_simd.h"
#include "extensions/YAKL_componentwise.h"
#include "extensions/YAKL_intrinsics.h"

