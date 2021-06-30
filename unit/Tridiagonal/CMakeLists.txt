

set(MYSRC Tridiagonal.cpp)

add_executable(Tridiagonal ${MYSRC})
target_compile_features(Tridiagonal PUBLIC cxx_std_14)
include(../../process_cxx_source_files.cmake)
process_cxx_source_files(${MYSRC})
target_link_libraries(Tridiagonal yakl)

add_test(NAME Tridiagonal_test COMMAND ./Tridiagonal)

