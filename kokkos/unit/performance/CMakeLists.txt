
set(MYSRC performance.cpp)
add_executable(performance ${MYSRC})
target_link_libraries(performance yakl kokkos)
target_compile_options(performance PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${YAKL_UNIT_CXX_FLAGS}>)
if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
  target_link_libraries(performance "${YAKL_UNIT_CXX_LINK_FLAGS}")
endif()
separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})
# add_test(NAME performance_test COMMAND ${REAL_MPI_COMMAND} ./performance)

