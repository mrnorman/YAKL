

set(MYSRC Reductions.cpp)

add_executable(Reductions ${MYSRC})
include(../../yakl_utils.cmake)
yakl_process_target(Reductions)

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME Reductions_test COMMAND ${REAL_MPI_COMMAND} ./Reductions)

