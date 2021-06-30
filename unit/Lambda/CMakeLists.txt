

set(MYSRC Lambda.cpp)

add_executable(Lambda ${MYSRC})
target_compile_features(Lambda PUBLIC cxx_std_14)
include(../../process_cxx_source_files.cmake)
process_cxx_source_files(${MYSRC})
target_link_libraries(Lambda yakl)

add_test(NAME Lambda_test COMMAND ./Lambda)

