

set(MYSRC CArray.cpp)

add_executable(CArray ${MYSRC})
target_compile_features(CArray PUBLIC cxx_std_14)
include(../../yakl_utils.cmake)
yakl_process_target(CArray)

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME CArray_test COMMAND ${REAL_MPI_COMMAND} ./CArray)

