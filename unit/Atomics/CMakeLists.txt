
set(MYSRC Atomics.cpp)
add_executable(Atomics ${MYSRC})
target_link_libraries(Atomics yakl kokkos)
target_compile_options(Atomics PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${YAKL_UNIT_CXX_FLAGS}>)
if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
  target_link_libraries(Atomics "${YAKL_UNIT_CXX_LINK_FLAGS}")
endif()
separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})
add_test(NAME Atomics_test COMMAND ${REAL_MPI_COMMAND} ./Atomics)

