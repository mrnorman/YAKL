

set(MYSRC Intrinsics.cpp)

add_executable(Intrinsics ${MYSRC})
include(../../yakl_utils.cmake)
yakl_process_target(Intrinsics)

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME Intrinsics_test COMMAND ${REAL_MPI_COMMAND} ./Intrinsics)

