
set(MYSRC PNetCDF.cpp)
add_executable(PNetCDF ${MYSRC})
target_link_libraries(PNetCDF yakl)
target_compile_options(PNetCDF PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${YAKL_UNIT_CXX_FLAGS}>)
if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
  target_link_libraries(PNetCDF "${YAKL_UNIT_CXX_LINK_FLAGS}")
endif()
separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})
add_test(NAME PNetCDF_test COMMAND ${REAL_MPI_COMMAND} ./PNetCDF)

