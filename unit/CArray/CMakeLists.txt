

set(MYSRC CArray.cpp)

add_executable(CArray ${MYSRC})
include(../../yakl_utils.cmake)
yakl_process_target(CArray)

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME CArray_test COMMAND ${REAL_MPI_COMMAND} ./CArray)

