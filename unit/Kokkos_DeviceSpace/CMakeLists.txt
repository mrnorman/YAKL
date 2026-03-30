
set(MYSRC Kokkos_DeviceSpace.cpp)
add_executable(Kokkos_DeviceSpace ${MYSRC})
target_link_libraries(Kokkos_DeviceSpace yakl)
target_compile_options(Kokkos_DeviceSpace PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${YAKL_UNIT_CXX_FLAGS}>)
if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
  target_link_libraries(Kokkos_DeviceSpace "${YAKL_UNIT_CXX_LINK_FLAGS}")
endif()
separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})
add_test(NAME Kokkos_DeviceSpace_test COMMAND ${REAL_MPI_COMMAND} ./Kokkos_DeviceSpace)

