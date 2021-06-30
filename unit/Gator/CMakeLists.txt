

set(MYSRC TestGator.cpp)

add_executable(Gator ${MYSRC})
target_compile_features(Gator PUBLIC cxx_std_14)
include(../../process_cxx_source_files.cmake)
process_cxx_source_files(${MYSRC})
target_link_libraries(Gator yakl)

add_test(NAME Gator_test COMMAND ./Gator)

