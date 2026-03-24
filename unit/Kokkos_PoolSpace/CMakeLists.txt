
set(MYSRC Kokkos_PoolSpace.cpp)
add_executable(Kokkos_PoolSpace ${MYSRC})
target_link_libraries(Kokkos_PoolSpace yakl)
target_compile_options(Kokkos_PoolSpace PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${YAKL_UNIT_CXX_FLAGS}>)
if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
  target_link_libraries(Kokkos_PoolSpace "${YAKL_UNIT_CXX_LINK_FLAGS}")
endif()
separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})
add_test(NAME Kokkos_PoolSpace_test COMMAND ${REAL_MPI_COMMAND} ./Kokkos_PoolSpace)

