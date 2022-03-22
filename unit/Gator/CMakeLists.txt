

set(MYSRC TestGator.cpp)

add_executable(Gator ${MYSRC})
include(../../yakl_utils.cmake)
yakl_process_target(Gator)

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME Gator_test COMMAND ${REAL_MPI_COMMAND} ./Gator)

