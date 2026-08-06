
add_executable(Fortran_Gator ${CMAKE_CURRENT_SOURCE_DIR}/Fortran_Gator.F90)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../yakl)
set_target_properties(Fortran_Gator PROPERTIES LINKER_LANGUAGE Fortran)
target_link_libraries(Fortran_Gator yakl stdc++)
separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})
add_test(NAME Fortran_Gator_test COMMAND ${REAL_MPI_COMMAND} ./Fortran_Gator)

add_executable(Fortran_Gator_Validation Fortran_Gator_Validation.F90)
set_target_properties(Fortran_Gator_Validation PROPERTIES LINKER_LANGUAGE Fortran)
target_link_libraries(Fortran_Gator_Validation yakl stdc++)
target_compile_options(Fortran_Gator_Validation PUBLIC $<$<COMPILE_LANGUAGE:Fortran>:${YAKL_F90_FLAGS}>)
add_test(NAME Fortran_Gator_bytes_positive COMMAND ./Fortran_Gator_Validation positive)
foreach(SCENARIO zero negative overflow)
  add_test(NAME Fortran_Gator_bytes_${SCENARIO}
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/../Validation/expect_failure.sh
                        $<TARGET_FILE:Fortran_Gator_Validation> ${SCENARIO})
endforeach()
