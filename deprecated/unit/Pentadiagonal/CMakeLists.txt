

set(MYSRC Pentadiagonal.cpp)

add_executable(Pentadiagonal ${MYSRC})
include(../../yakl_utils.cmake)
yakl_process_target(Pentadiagonal)

if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
  target_link_libraries(Pentadiagonal "${YAKL_UNIT_CXX_LINK_FLAGS}")
endif()

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME Pentadiagonal_test COMMAND ${REAL_MPI_COMMAND} ./Pentadiagonal)

