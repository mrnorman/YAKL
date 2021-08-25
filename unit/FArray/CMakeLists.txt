

set(MYSRC FArray.cpp)

add_executable(FArray ${MYSRC})
target_compile_features(FArray PUBLIC cxx_std_14)
include(../../yakl_utils.cmake)
yakl_process_target(FArray)

add_test(NAME FArray_test COMMAND ${MPI_COMMAND} ./FArray)

