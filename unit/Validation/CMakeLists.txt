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
foreach(SCENARIO array_slice array_subset array_reshape component_shape sign_shape merge_value_shape
                 merge_condition_shape loop_extent loop_stride simple_bounds_negative simple_bounds_overflow
                 linear_allocator linear_allocator_zero_pool linear_allocator_small_block linear_allocator_overflow
                 linear_allocator_allocation_overflow linear_allocator_exhaustion linear_allocator_invalid_free
                 linear_allocator_double_free linear_allocator_end_pointer linear_allocator_uninitialized
                 linear_allocator_empty_callback matinv_singular matinv_near_singular random_range timer_stop
                 unpack_index autotune_index allocate_before_init free_before_init finalize_with_live_allocation)
  add_test(NAME Validation_${SCENARIO}
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.sh $<TARGET_FILE:Validation> ${SCENARIO})
endforeach()
