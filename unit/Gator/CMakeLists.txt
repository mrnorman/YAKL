

set(MYSRC TestGator.cpp)

add_executable(Gator ${MYSRC})
include(../../yakl_utils.cmake)
yakl_process_target(Gator)

if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
  target_link_libraries(Gator "${YAKL_UNIT_CXX_LINK_FLAGS}")
endif()

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME Gator_test COMMAND ${REAL_MPI_COMMAND} ./Gator)

