

set(MYSRC Tridiagonal.cpp)

add_executable(Tridiagonal ${MYSRC})
include(../../yakl_utils.cmake)
yakl_process_target(Tridiagonal)

if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
  target_link_libraries(Tridiagonal "${YAKL_UNIT_CXX_LINK_FLAGS}")
endif()

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME Tridiagonal_test COMMAND ${REAL_MPI_COMMAND} ./Tridiagonal)

