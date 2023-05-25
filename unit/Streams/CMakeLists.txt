

set(MYSRC Streams.cpp)

add_executable(Streams ${MYSRC})
include(../../yakl_utils.cmake)
yakl_process_target(Streams)

if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
  target_link_libraries(Streams "${YAKL_UNIT_CXX_LINK_FLAGS}")
endif()

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME Streams_test COMMAND ${REAL_MPI_COMMAND} ./Streams)

