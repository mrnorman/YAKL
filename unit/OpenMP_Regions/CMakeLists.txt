

set(MYSRC OpenMP_Regions.cpp)

add_executable(OpenMP_Regions ${MYSRC})
include(../../yakl_utils.cmake)
yakl_process_target(OpenMP_Regions)

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME OpenMP_Regions_test COMMAND ${REAL_MPI_COMMAND} ./OpenMP_Regions)

