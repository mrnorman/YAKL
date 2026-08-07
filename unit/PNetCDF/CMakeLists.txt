
set(MYSRC PNetCDF.cpp)
add_executable(PNetCDF ${MYSRC})
target_link_libraries(PNetCDF yakl)
target_compile_options(PNetCDF PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${YAKL_UNIT_CXX_FLAGS}>)
if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
  target_link_libraries(PNetCDF "${YAKL_UNIT_CXX_LINK_FLAGS}")
endif()
separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})
add_test(NAME PNetCDF_test COMMAND ${REAL_MPI_COMMAND} ./PNetCDF)
add_test(NAME PNetCDF_dimension_mismatch_test
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/../Validation/expect_failure.sh
                      ${REAL_MPI_COMMAND} $<TARGET_FILE:PNetCDF> dimension_mismatch)
add_test(NAME PNetCDF_offset_overflow_test
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/../Validation/expect_failure.sh
                      ${REAL_MPI_COMMAND} $<TARGET_FILE:PNetCDF> offset_overflow)
