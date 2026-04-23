
#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ProfileSection.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include "YAKL_header.h"
#include "YAKL_defines.h"
#include "YAKL_LinearAllocator.h"
#include "YAKL_Toney.h"
#include "YAKL_Internal.h"
#include "YAKL_timers.h"
#include "YAKL_allocators.h"
#include "YAKL_Kokkos_DeviceSpace.h"
#include "YAKL_InitConfig.h"
#include "YAKL_init.h"
#include "YAKL_parallel_for.h"
#include "YAKL_parallel_for_autotune.h"
#include "YAKL_finalize.h"
#include "YAKL_random.h"
#include "YAKL_SArray.h"
#include "YAKL_SArray_F.h"
#include "YAKL_Array.h"
#include "YAKL_Array_F.h"
#include "YAKL_ScalarLiveOut.h"
#include "YAKL_componentwise.h"
#include "YAKL_intrinsics.h"

