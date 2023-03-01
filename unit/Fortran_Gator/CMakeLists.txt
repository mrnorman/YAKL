

configure_file(Fortran_Gator.F90.in Fortran_Gator${YAKL_NAME_APPEND}.F90)
add_executable(Fortran_Gator ${CMAKE_CURRENT_BINARY_DIR}/Fortran_Gator${YAKL_NAME_APPEND}.F90)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../yakl)
set_target_properties(Fortran_Gator PROPERTIES LINKER_LANGUAGE Fortran)
include(../../yakl_utils.cmake)
yakl_process_target(Fortran_Gator)
target_link_libraries(Fortran_Gator ${YAKL_FORTRAN_INTERFACE_TARGET})


separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME Fortran_Gator_test COMMAND ${REAL_MPI_COMMAND} ./Fortran_Gator)

