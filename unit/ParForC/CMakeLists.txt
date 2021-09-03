

set(MYSRC ParForC.cpp)

add_executable(ParForC ${MYSRC})
target_compile_features(ParForC PUBLIC cxx_std_14)
include(../../yakl_utils.cmake)
yakl_process_target(ParForC)

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME ParForC_test COMMAND ${REAL_MPI_COMMAND} ./ParForC)

