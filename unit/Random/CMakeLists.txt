

set(MYSRC Random.cpp)

add_executable(Random ${MYSRC})
target_compile_features(Random PUBLIC cxx_std_14)
include(../../yakl_utils.cmake)
yakl_process_target(Random)

add_test(NAME Random_test COMMAND ${MPI_COMMAND} ./Random)

