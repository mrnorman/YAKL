set(MYSRC Validation.cpp)
add_executable(Validation ${MYSRC})
target_link_libraries(Validation yakl)
target_compile_options(Validation PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${YAKL_UNIT_CXX_FLAGS}>)
if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
  target_link_libraries(Validation "${YAKL_UNIT_CXX_LINK_FLAGS}")
endif()

add_test(NAME Validation_positive COMMAND ./Validation positive)
add_test(NAME Validation_environment_negative COMMAND ./Validation environment_negative)
add_test(NAME Validation_config_disable COMMAND ./Validation config_disable)
add_test(NAME Validation_config_enable COMMAND ./Validation config_enable)
add_test(NAME Validation_config_size_default COMMAND ./Validation config_size_default)
add_test(NAME Validation_config_block_precedence COMMAND ./Validation config_block_precedence)
add_test(NAME Validation_lifecycle_reset COMMAND ./Validation lifecycle_reset)
set(YAKL_ALWAYS_FAILURE_SCENARIOS
                 array_reshape linear_allocator linear_allocator_zero_pool linear_allocator_small_block linear_allocator_overflow
                 linear_allocator_allocation_overflow linear_allocator_exhaustion linear_allocator_invalid_free
                 linear_allocator_double_free linear_allocator_uninitialized linear_allocator_empty_callback timer_stop
                 allocate_before_init free_before_init
                 finalize_with_live_allocation)

set(YAKL_DEBUG_FAILURE_SCENARIOS
                 array_f_bound_overflow component_shape sign_shape merge_value_shape merge_condition_shape
                 loop_extent loop_stride loop_f_extent loop_f_stride simple_bounds_negative simple_bounds_overflow
                 matinv_singular matinv_near_singular random_range random_normal_stddev random_bernoulli_probability
                 minloc_nan_sarray maxloc_nan_sarray minloc_nan_array maxloc_nan_array
                 random_exponential_rate random_lognormal_stddev
                 random_uniform_nan_lower random_uniform_nan_upper random_normal_nan_mean random_normal_nan_stddev
                 random_bernoulli_nan random_exponential_nan random_lognormal_nan_stddev
                 autotune_index config_tile_zero config_tile_negative)

set(YAKL_BOUNDS_DEBUG_FAILURE_SCENARIOS array_slice array_subset linear_allocator_end_pointer unpack_index)

set(YAKL_ENABLED_FAILURE_SCENARIOS ${YAKL_ALWAYS_FAILURE_SCENARIOS})
if (Kokkos_ENABLE_DEBUG OR Kokkos_ENABLE_DEBUG_BOUNDS_CHECK)
  list(APPEND YAKL_ENABLED_FAILURE_SCENARIOS ${YAKL_DEBUG_FAILURE_SCENARIOS})
endif()
if (Kokkos_ENABLE_DEBUG_BOUNDS_CHECK)
  list(APPEND YAKL_ENABLED_FAILURE_SCENARIOS ${YAKL_BOUNDS_DEBUG_FAILURE_SCENARIOS})
endif()
if (YAKL_INDEX_BITS STREQUAL "32")
  list(APPEND YAKL_ENABLED_FAILURE_SCENARIOS index_width_overflow simple_bounds_overflow)
  list(REMOVE_DUPLICATES YAKL_ENABLED_FAILURE_SCENARIOS)
endif()

foreach(SCENARIO ${YAKL_ENABLED_FAILURE_SCENARIOS})
  add_test(NAME Validation_${SCENARIO}
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.sh $<TARGET_FILE:Validation> ${SCENARIO})
endforeach()
