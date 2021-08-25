

set(MYSRC TestGator.cpp)

add_executable(Gator ${MYSRC})
target_compile_features(Gator PUBLIC cxx_std_14)
include(../../yakl_utils.cmake)
yakl_process_target(Gator)

add_test(NAME Gator_test COMMAND ${MPI_COMMAND} ./Gator)

