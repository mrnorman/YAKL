

set(MYSRC ParForFortran.cpp)

add_executable(ParForFortran ${MYSRC})
target_compile_features(ParForFortran PUBLIC cxx_std_14)
include(../../yakl_utils.cmake)
yakl_process_target(ParForFortran)

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME ParForFortran_test COMMAND ${REAL_MPI_COMMAND} ./ParForFortran)

