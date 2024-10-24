
configure_file(wrapper.F90.in wrapper${YAKL_NAME_APPEND}.F90)
add_executable(Fortran_Link simple.cpp ${CMAKE_CURRENT_BINARY_DIR}/wrapper${YAKL_NAME_APPEND}.F90)
target_link_libraries(Fortran_Link yakl_fortran yakl kokkos)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../yakl)
set_target_properties(Fortran_Link PROPERTIES LINKER_LANGUAGE Fortran)
separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})
add_test(NAME Fortran_Link_test COMMAND ${REAL_MPI_COMMAND} ./Fortran_Link)

