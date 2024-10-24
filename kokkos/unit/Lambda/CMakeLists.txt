
set(MYSRC Lambda.cpp)
add_executable(Lambda ${MYSRC})
target_link_libraries(Lambda yakl kokkos)
target_compile_options(Lambda PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${YAKL_UNIT_CXX_FLAGS}>)
if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
  target_link_libraries(Lambda "${YAKL_UNIT_CXX_LINK_FLAGS}")
endif()
separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})
add_test(NAME Lambda_test COMMAND ${REAL_MPI_COMMAND} ./Lambda)

