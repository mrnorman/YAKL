

set(MYSRC FArray.cpp)

add_executable(FArray ${MYSRC})
include(../../yakl_utils.cmake)
yakl_process_target(FArray)

if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
  target_link_libraries(FArray "${YAKL_UNIT_CXX_LINK_FLAGS}")
endif()

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME FArray_test COMMAND ${REAL_MPI_COMMAND} ./FArray)

