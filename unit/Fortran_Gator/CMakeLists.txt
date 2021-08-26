

set(F90_SRC Fortran_Gator.F90)

add_executable(Fortran_Gator ${F90_SRC})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../yakl)
set_target_properties(Fortran_Gator PROPERTIES LINKER_LANGUAGE Fortran)
include(../../yakl_utils.cmake)
yakl_process_target(Fortran_Gator)

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME Fortran_Gator_test COMMAND ${REAL_MPI_COMMAND} ./Fortran_Gator)

