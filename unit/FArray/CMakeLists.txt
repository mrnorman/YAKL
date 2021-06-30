

set(MYSRC FArray.cpp)

add_executable(FArray ${MYSRC})
target_compile_features(FArray PUBLIC cxx_std_14)
include(../../process_cxx_source_files.cmake)
process_cxx_source_files(${MYSRC})
target_link_libraries(FArray yakl)

add_test(NAME FArray_test COMMAND ./FArray)

