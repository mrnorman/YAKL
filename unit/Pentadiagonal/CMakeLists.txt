

set(MYSRC Pentadiagonal.cpp)

add_executable(Pentadiagonal ${MYSRC})
target_compile_features(Pentadiagonal PUBLIC cxx_std_14)
include(../../yakl_utils.cmake)
yakl_process_target(Pentadiagonal)

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME Pentadiagonal_test COMMAND ${REAL_MPI_COMMAND} ./Pentadiagonal)

