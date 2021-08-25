

set(MYSRC CArray.cpp)

add_executable(CArray ${MYSRC})
target_compile_features(CArray PUBLIC cxx_std_14)
include(../../yakl_utils.cmake)
yakl_process_target(CArray)

add_test(NAME CArray_test COMMAND ${MPI_COMMAND} ./CArray)

