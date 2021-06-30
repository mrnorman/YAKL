

set(MYSRC Pentadiagonal.cpp)

add_executable(Pentadiagonal ${MYSRC})
target_compile_features(Pentadiagonal PUBLIC cxx_std_14)
include(../../process_cxx_source_files.cmake)
process_cxx_source_files(${MYSRC})
target_link_libraries(Pentadiagonal yakl)

add_test(NAME Pentadiagonal_test COMMAND ./Pentadiagonal)

