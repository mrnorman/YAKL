

set(MYSRC Atomics.cpp)

add_executable(Atomics ${MYSRC})
include(../../yakl_utils.cmake)
yakl_process_target(Atomics)

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME Atomics_test COMMAND ${REAL_MPI_COMMAND} ./Atomics)

