
set(MYSRC OpenMP_Regions.cpp)
add_executable(OpenMP_Regions ${MYSRC})
target_link_libraries(OpenMP_Regions yakl kokkos)
target_compile_options(OpenMP_Regions PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${YAKL_UNIT_CXX_FLAGS}>)
if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
  target_link_libraries(OpenMP_Regions "${YAKL_UNIT_CXX_LINK_FLAGS}")
endif()
separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})
add_test(NAME OpenMP_Regions_test COMMAND ${REAL_MPI_COMMAND} ./OpenMP_Regions)

