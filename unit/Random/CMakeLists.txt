

set(MYSRC Random.cpp)

add_executable(Random ${MYSRC})
target_compile_features(Random PUBLIC cxx_std_14)
include(../../process_cxx_source_files.cmake)
process_cxx_source_files(${MYSRC})
target_link_libraries(Random yakl)

add_test(NAME Random_test COMMAND ./Random)

