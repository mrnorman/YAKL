

set(MYSRC Lambda.cpp)

add_executable(Lambda ${MYSRC})
include(../../yakl_utils.cmake)
yakl_process_target(Lambda)

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME Lambda_test COMMAND ${REAL_MPI_COMMAND} ./Lambda)

