
set(MYSRC NetCDF.cpp)
add_executable(NetCDF ${MYSRC})
target_link_libraries(NetCDF yakl kokkos)
target_compile_options(NetCDF PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${YAKL_UNIT_CXX_FLAGS}>)
if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
  target_link_libraries(NetCDF "${YAKL_UNIT_CXX_LINK_FLAGS}")
endif()
separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})
add_test(NAME NetCDF_test COMMAND ${REAL_MPI_COMMAND} ./NetCDF)

