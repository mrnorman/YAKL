
set(MYSRC ParForC.cpp)
add_executable(ParForC ${MYSRC})
target_link_libraries(ParForC yakl kokkos)
target_compile_options(ParForC PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${YAKL_UNIT_CXX_FLAGS}>)
if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
  target_link_libraries(ParForC "${YAKL_UNIT_CXX_LINK_FLAGS}")
endif()
separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})
add_test(NAME ParForC_test COMMAND ${REAL_MPI_COMMAND} ./ParForC)

