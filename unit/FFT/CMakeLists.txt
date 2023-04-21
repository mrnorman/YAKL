

set(MYSRC FFT.cpp)

add_executable(FFT ${MYSRC})
include(../../yakl_utils.cmake)
yakl_process_target(FFT)

if (NOT "${YAKL_UNIT_CXX_LINK_FLAGS}" STREQUAL "")
  target_link_libraries(FFT "${YAKL_UNIT_CXX_LINK_FLAGS}")
endif()

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME FFT_test COMMAND ${REAL_MPI_COMMAND} ./FFT)

