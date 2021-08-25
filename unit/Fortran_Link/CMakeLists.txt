

set(CXX_SRC simple.cpp)
set(F90_SRC wrapper.F90)

add_executable(Fortran_Link ${CXX_SRC} ${F90_SRC})
target_compile_features(Fortran_Link PUBLIC cxx_std_14)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../yakl)
target_link_libraries(Fortran_Link yakl)
set_target_properties(Fortran_Link PROPERTIES LINKER_LANGUAGE Fortran)
include(../../yakl_utils.cmake)
yakl_process_cxx_source_files(${CXX_SRC})

separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND ${MPI_COMMAND})

add_test(NAME Fortran_Link_test COMMAND ${REAL_MPI_COMMAND} ./Fortran_Link)

