

set(MYSRC Reductions.cpp)

add_executable(Reductions ${MYSRC})
include(../../yakl_utils.cmake)
yakl_process_target(Reductions)

if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
  target_link_libraries(Reductions "${YAKL_UNIT_CXX_LINK_FLAGS}")
endif()

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME Reductions_test COMMAND ${REAL_MPI_COMMAND} ./Reductions)

