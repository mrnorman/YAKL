
add_executable(Fortran_Gator ${CMAKE_CURRENT_SOURCE_DIR}/Fortran_Gator.F90)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../yakl)
set_target_properties(Fortran_Gator PROPERTIES LINKER_LANGUAGE Fortran)
target_link_libraries(Fortran_Gator yakl_fortran yakl kokkos)
separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})
add_test(NAME Fortran_Gator_test COMMAND ${REAL_MPI_COMMAND} ./Fortran_Gator)

