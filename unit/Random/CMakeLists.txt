

set(MYSRC Random.cpp)

add_executable(Random ${MYSRC})
target_compile_features(Random PUBLIC cxx_std_14)
include(../../yakl_utils.cmake)
yakl_process_target(Random)

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME Random_test COMMAND ${REAL_MPI_COMMAND} ./Random)

