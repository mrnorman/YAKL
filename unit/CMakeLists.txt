cmake_minimum_required(VERSION 3.22)
project(yakl_unit)
enable_testing()
option(YAKL_UNIT_LARGE_MEMORY "Build tests requiring more than 4 GB of device memory" OFF)
if (YAKL_INDEX_BITS STREQUAL "32" AND YAKL_UNIT_LARGE_MEMORY)
  message(STATUS "Disabling YAKL_UNIT_LARGE_MEMORY because YAKL_INDEX_BITS=32")
  set(YAKL_UNIT_LARGE_MEMORY OFF CACHE BOOL "Build tests requiring more than 4 GB of device memory" FORCE)
endif()

enable_language(Fortran)
enable_language(CXX)
enable_language(C)

set(CMAKE_CXX_STANDARD 20)

set(Kokkos_HOME "" CACHE PATH "Path to a Kokkos source tree")
if (Kokkos_HOME STREQUAL "")
  message(FATAL_ERROR "Kokkos_HOME must point to a Kokkos source tree")
endif()
get_filename_component(Kokkos_HOME "${Kokkos_HOME}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
if (NOT EXISTS "${Kokkos_HOME}/CMakeLists.txt")
  message(FATAL_ERROR "Kokkos_HOME does not contain a Kokkos source tree: ${Kokkos_HOME}")
endif()
message(STATUS "YAKL unit tests are using Kokkos from ${Kokkos_HOME}")
set(Kokkos_BIN  ${CMAKE_CURRENT_BINARY_DIR}/kokkos)
add_subdirectory(${Kokkos_HOME} ${Kokkos_BIN})

set(YAKL_HOME ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(YAKL_BIN  ${CMAKE_CURRENT_BINARY_DIR}/yakl)
add_subdirectory(${YAKL_HOME} ${YAKL_BIN})

# Unit test directories
add_subdirectory(Atomics)
add_subdirectory(CArray)
add_subdirectory(FArray)
add_subdirectory(Fortran_Gator)
add_subdirectory(Fortran_Link)
add_subdirectory(Intrinsics)
add_subdirectory(Kokkos_DeviceSpace)
add_subdirectory(Lambda)
if (YAKL_TEST_NETCDF)
  add_subdirectory(NetCDF)
endif()
if (YAKL_TEST_PNETCDF)
  add_subdirectory(PNetCDF)
endif()
add_subdirectory(OpenMP_Regions)
add_subdirectory(ParForC)
add_subdirectory(ParForFortran)
add_subdirectory(performance EXCLUDE_FROM_ALL)
add_subdirectory(Random)
add_subdirectory(Reductions)
add_subdirectory(Timers)
add_subdirectory(Validation)
if (YAKL_UNIT_LARGE_MEMORY)
  add_subdirectory(LargeMemory)
endif()

if (YAKL_ENABLE_COVERAGE)
  find_program(YAKL_GCOV_EXECUTABLE gcov)
  find_program(YAKL_GCOVR_EXECUTABLE gcovr)
  if (NOT YAKL_GCOV_EXECUTABLE)
    message(FATAL_ERROR "YAKL_ENABLE_COVERAGE requires gcov")
  endif()
  if (NOT YAKL_GCOVR_EXECUTABLE)
    message(FATAL_ERROR "YAKL_ENABLE_COVERAGE requires gcovr")
  endif()

  set(YAKL_UNIT_TESTS Atomics_test
                      CArray_test
                      FArray_test
                      Fortran_Gator_test
                      Fortran_Gator_wrappers
                      Fortran_Gator_wrappers_no_pool
                      Fortran_Gator_kokkos_ownership
                      Fortran_Gator_kokkos_yakl_ownership
                      Fortran_Gator_bytes_positive
                      Fortran_Gator_bytes_zero
                      Fortran_Gator_bytes_negative
                      Fortran_Gator_bytes_overflow
                      Fortran_Gator_associated_allocate
                      Fortran_Gator_disassociated_deallocate
                      Fortran_Gator_interior_deallocate
                      Fortran_Gator_zero_extent
                      Fortran_Gator_lower_bound_overflow
                      Fortran_Link_test
                      Intrinsics_test
                      Kokkos_DeviceSpace_test
                      Lambda_test
                      OpenMP_Regions_test
                      ParForC_test
                      ParForFortran_test
                      Random_test
                      Reductions_test
                      Timers_disabled_test
                      Timers_profile_test
                      Timers_auto_profile_test
                      Validation_positive
                      Validation_environment_negative
                      Validation_config_disable
                      Validation_config_enable
                      Validation_config_size_default
                      Validation_config_block_precedence
                      Validation_lifecycle_reset
                      Validation_array_f_bound_overflow
                      Validation_array_slice
                      Validation_array_subset
                      Validation_array_reshape
                      Validation_component_shape
                      Validation_sign_shape
                      Validation_merge_value_shape
                      Validation_merge_condition_shape
                      Validation_loop_extent
                      Validation_loop_stride
                      Validation_loop_f_extent
                      Validation_loop_f_stride
                      Validation_simple_bounds_negative
                      Validation_simple_bounds_overflow
                      Validation_linear_allocator
                      Validation_linear_allocator_zero_pool
                      Validation_linear_allocator_small_block
                      Validation_linear_allocator_overflow
                      Validation_linear_allocator_allocation_overflow
                      Validation_linear_allocator_exhaustion
                      Validation_linear_allocator_invalid_free
                      Validation_linear_allocator_double_free
                      Validation_linear_allocator_end_pointer
                      Validation_linear_allocator_uninitialized
                      Validation_linear_allocator_empty_callback
                      Validation_matinv_singular
                      Validation_matinv_near_singular
                      Validation_minloc_nan_sarray
                      Validation_maxloc_nan_sarray
                      Validation_minloc_nan_array
                      Validation_maxloc_nan_array
                      Validation_random_range
                      Validation_random_normal_stddev
                      Validation_random_bernoulli_probability
                      Validation_random_exponential_rate
                      Validation_random_lognormal_stddev
                      Validation_random_uniform_nan_lower
                      Validation_random_uniform_nan_upper
                      Validation_random_normal_nan_mean
                      Validation_random_normal_nan_stddev
                      Validation_random_bernoulli_nan
                      Validation_random_exponential_nan
                      Validation_random_lognormal_nan_stddev
                      Validation_timer_stop
                      Validation_unpack_index
                      Validation_autotune_index
                      Validation_config_tile_zero
                      Validation_config_tile_negative
                      Validation_allocate_before_init
                      Validation_free_before_init
                      Validation_finalize_with_live_allocation)
  if (NOT Kokkos_ENABLE_DEBUG AND NOT Kokkos_ENABLE_DEBUG_BOUNDS_CHECK)
    list(REMOVE_ITEM YAKL_UNIT_TESTS
                     Validation_component_shape
                     Validation_sign_shape
                     Validation_merge_value_shape
                     Validation_merge_condition_shape
                     Validation_loop_extent
                     Validation_loop_stride
                     Validation_loop_f_extent
                     Validation_loop_f_stride
                     Validation_simple_bounds_negative
                     Validation_simple_bounds_overflow
                     Validation_array_f_bound_overflow
                     Validation_matinv_singular
                     Validation_matinv_near_singular
                     Validation_minloc_nan_sarray
                     Validation_maxloc_nan_sarray
                     Validation_minloc_nan_array
                     Validation_maxloc_nan_array
                     Validation_random_range
                     Validation_random_normal_stddev
                     Validation_random_bernoulli_probability
                     Validation_random_exponential_rate
                     Validation_random_lognormal_stddev
                     Validation_random_uniform_nan_lower
                     Validation_random_uniform_nan_upper
                     Validation_random_normal_nan_mean
                     Validation_random_normal_nan_stddev
                     Validation_random_bernoulli_nan
                     Validation_random_exponential_nan
                     Validation_random_lognormal_nan_stddev
                     Validation_autotune_index
                     Validation_config_tile_zero
                     Validation_config_tile_negative)
  endif()
  if (NOT Kokkos_ENABLE_DEBUG_BOUNDS_CHECK)
    list(REMOVE_ITEM YAKL_UNIT_TESTS
                     Validation_array_slice
                     Validation_array_subset
                     Validation_linear_allocator_end_pointer
                     Validation_unpack_index)
  endif()
  if (YAKL_TEST_NETCDF)
    list(APPEND YAKL_UNIT_TESTS NetCDF_test NetCDF_dimension_mismatch_test)
  endif()
  if (YAKL_TEST_PNETCDF)
    list(APPEND YAKL_UNIT_TESTS PNetCDF_test PNetCDF_dimension_mismatch_test PNetCDF_offset_overflow_test)
  endif()
  if (YAKL_UNIT_LARGE_MEMORY)
    list(APPEND YAKL_UNIT_TESTS LargeMemory_test)
  endif()

  add_test(NAME zz_coverage_summary
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/coverage_summary.sh
                        ${YAKL_HOME}
                        ${CMAKE_CURRENT_BINARY_DIR}
                        ${YAKL_GCOVR_EXECUTABLE}
                        ${YAKL_GCOV_EXECUTABLE})
  set_tests_properties(zz_coverage_summary PROPERTIES DEPENDS "${YAKL_UNIT_TESTS}")
endif()
