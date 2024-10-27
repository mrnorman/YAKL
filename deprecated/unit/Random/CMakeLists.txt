

set(MYSRC Random.cpp)

add_executable(Random ${MYSRC})
include(../../yakl_utils.cmake)
yakl_process_target(Random)

if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
  target_link_libraries(Random "${YAKL_UNIT_CXX_LINK_FLAGS}")
endif()

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME Random_test COMMAND ${REAL_MPI_COMMAND} ./Random)

